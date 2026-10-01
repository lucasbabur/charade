import numpy as np
import polars as pl

from charade.features.spec import CATEGORICAL_FIELDS, FeatureSpec, Group, encode, fit_spec


def _fit(features: pl.DataFrame) -> FeatureSpec:
    return fit_spec(features.filter(pl.col("split") == "train"), set(Group) - {Group.TEXT})


def test_vocabulary_uses_train_only(features: pl.DataFrame) -> None:
    spec = _fit(features)
    test_only = set(features.filter(pl.col("split") == "test")["C14"].unique().to_list()) - set(
        features.filter(pl.col("split") == "train")["C14"].unique().to_list()
    )
    assert test_only
    assert not test_only & set(spec.vocabularies["C14"])


def test_unseen_values_encode_to_oov(features: pl.DataFrame) -> None:
    spec = _fit(features)
    frame = features.head(3).with_columns(pl.lit("never-seen").alias("C14"))
    assert (encode(spec, frame).categorical[:, spec.field_index("C14")] == 0).all()


def test_dense_is_standardized_on_train(features: pl.DataFrame) -> None:
    spec = _fit(features)
    dense = encode(spec, features.filter(pl.col("split") == "train")).dense
    np.testing.assert_allclose(dense.mean(axis=0), 0, atol=1e-4)


def test_dropping_a_group_removes_its_fields(features: pl.DataFrame) -> None:
    spec = fit_spec(features.filter(pl.col("split") == "train"), {Group.CONTEXT, Group.AD})
    assert {f.group for f in spec.categorical} == {Group.CONTEXT, Group.AD}
    assert spec.dense == []


def test_spec_roundtrips_through_json(features: pl.DataFrame) -> None:
    spec = _fit(features)
    restored = FeatureSpec.model_validate_json(spec.model_dump_json())
    np.testing.assert_array_equal(encode(spec, features).categorical, encode(restored, features).categorical)


def test_every_declared_field_is_derived(features: pl.DataFrame) -> None:
    assert {f.name for f in CATEGORICAL_FIELDS} <= set(features.columns)
