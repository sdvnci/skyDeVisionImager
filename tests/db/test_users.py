import pytest
from flask.testing import FlaskClient
from pydantic import SecretStr

from server.db import (
    AlreadyExistsError,
    InvalidInputError,
    Manager,
    NotFoundError,
    User,
    UserRole,
    UserService,
)


def make_user(n: int = 1, role: UserRole = UserRole.MEMBER) -> User:
    return User(
        ID=f"user-{n}",
        name=f"User {n}",
        email=f"user{n}@example.com",
        password=SecretStr(f"password-{n}"),
        role=role,
    )


def test_get_user_not_found(client: FlaskClient) -> None:
    assert UserService.get(id="nobody") is None
    assert UserService.get(email="nobody@example.com") is None


def test_get_user_requires_id_or_email(client: FlaskClient) -> None:
    with pytest.raises(InvalidInputError):
        _ = UserService.get()


def test_insert_and_get_user(client: FlaskClient) -> None:
    user = make_user(role=UserRole.ADMIN)
    UserService.insert(user)

    for stored in (UserService.get(id=user.ID), UserService.get(email=user.email)):
        assert stored is not None, "Could not retrieve newly inserted user"
        assert stored.ID == user.ID
        assert stored.name == user.name
        assert stored.email == user.email
        assert stored.role is UserRole.ADMIN


def test_user_password_is_stored_hashed(client: FlaskClient) -> None:
    user = make_user()
    UserService.insert(user)

    stored = UserService.get(id=user.ID)
    assert stored is not None
    stored_hash = stored.password.get_secret_value()
    assert stored_hash != user.password.get_secret_value()
    assert Manager.password_match(stored_hash, "password-1")
    assert not Manager.password_match(stored_hash, "not-the-password")


def test_long_passwords_are_not_truncated(client: FlaskClient) -> None:
    # bcrypt alone only reads the first 72 bytes
    password = "a" * 100
    hashed = Manager.hash_password(password)
    assert Manager.password_match(hashed, password)
    assert not Manager.password_match(hashed, "a" * 99)


def test_insert_user_duplicate(client: FlaskClient) -> None:
    UserService.insert(make_user())
    with pytest.raises(AlreadyExistsError):
        UserService.insert(make_user())


def test_insert_batch_is_all_or_nothing(client: FlaskClient) -> None:
    UserService.insert(make_user(2))
    with pytest.raises(AlreadyExistsError):
        UserService.insert_batch([make_user(1), make_user(2)])
    assert not UserService.exists(id="user-1"), "Failed batch left a user behind"


def test_list_users_paginates(client: FlaskClient) -> None:
    UserService.insert_batch([make_user(n) for n in range(1, 6)])

    users = UserService.list()
    assert [user.ID for user in users] == [f"user-{n}" for n in range(1, 6)]

    second_page = UserService.list(limit=2, page=1)
    assert [user.ID for user in second_page] == ["user-3", "user-4"]


def test_update_user(client: FlaskClient) -> None:
    user = make_user()
    UserService.insert(user)

    UserService.update(
        {"name": "Renamed", "role": UserRole.ADMIN, "password": SecretStr("changed")},
        id=user.ID,
    )

    stored = UserService.get(id=user.ID)
    assert stored is not None
    assert stored.name == "Renamed"
    assert stored.role is UserRole.ADMIN
    assert stored.email == user.email, "Fields not in the update must not change"
    assert Manager.password_match(stored.password.get_secret_value(), "changed")


def test_update_user_nonexistent(client: FlaskClient) -> None:
    with pytest.raises(NotFoundError):
        UserService.update({"name": "Renamed"}, id="nobody")


def test_update_user_rejects_unknown_columns(client: FlaskClient) -> None:
    UserService.insert(make_user())
    with pytest.raises(InvalidInputError):
        UserService.update(
            {"name = 'x'; DROP TABLE Users; --": "y"},  # type: ignore[typeddict-unknown-key]
            id="user-1",
        )
    assert UserService.exists(id="user-1")


def test_update_batch_is_all_or_nothing(client: FlaskClient) -> None:
    UserService.insert(make_user())
    with pytest.raises(NotFoundError):
        UserService.update_batch(
            [("user-1", None, {"name": "Renamed"}), ("nobody", None, {"name": "x"})]
        )

    stored = UserService.get(id="user-1")
    assert stored is not None
    assert stored.name == "User 1", "Failed batch left an update behind"


def test_delete_user(client: FlaskClient) -> None:
    user = make_user()
    UserService.insert(user)
    assert UserService.exists(email=user.email)

    UserService.delete(email=user.email)
    assert not UserService.exists(id=user.ID)

    with pytest.raises(NotFoundError):
        UserService.delete(id=user.ID)


def test_user_role_match() -> None:
    assert UserRole.match("admin") is UserRole.ADMIN
    assert UserRole.match("ADMIN") is UserRole.ADMIN
    assert UserRole.match("janitor") is UserRole.UNKNOWN
    assert UserRole("Member") is UserRole.MEMBER
