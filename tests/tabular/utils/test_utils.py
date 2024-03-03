import lightgbm as lgb
import pytest

from src.ml_trainer.tabular.utils.utils import generate_uid


@pytest.mark.parametrize(
    "args",
    [
        ("test",),  # from string
        (1,),  # from int
        (1, 2, 3),  # from multiple args
        ({"test": 1},),  # from dict
        (["test", 1],),  # from list
        (lambda x="dummy": x,),  # from function
        (
            {
                "callbacks": [
                    lgb.callback._EarlyStoppingCallback(
                        stopping_rounds=10,
                        verbose=True,
                    ),
                    lgb.callback._LogEvaluationCallback(
                        period=10,
                        show_stdv=True,
                    ),
                ]
            },
        ),
    ],
)
def test_can_generate_uid(args):  # noqa
    uid = generate_uid(*args)
    assert isinstance(uid, str)


@pytest.mark.parametrize(
    "args",
    [
        ("test",),
        (1,),
        (1, 2, 3),
        ({"test": 1},),
        (["test", 1],),
        (lambda x="dummy": x,),
        (
            {
                "callbacks": [
                    lgb.callback._EarlyStoppingCallback(
                        stopping_rounds=10,
                        verbose=True,
                    ),
                    lgb.callback._LogEvaluationCallback(
                        period=10,
                        show_stdv=True,
                    ),
                ]
            },
        ),
    ],
)
def test_generate_uid_is_consistent(args):  # noqa
    uid1 = generate_uid(*args)
    uid2 = generate_uid(*args)

    assert uid1 == uid2, "UIDs generated from the same inputs should be identical"
    print(uid1)
