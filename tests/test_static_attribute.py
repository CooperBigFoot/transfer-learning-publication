import logging

import pytest
import torch

from transfer_learning_publication.containers import StaticAttributeCollection


class TestStaticAttributeCollection:
    """Test StaticAttributeCollection class functionality."""

    @pytest.fixture
    def basic_data(self):
        """Create basic test data with 2 groups."""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),
        ]
        group_identifiers = ["group1", "group2"]
        attribute_names = ["attr_a", "attr_b", "attr_c"]
        return tensors, attribute_names, group_identifiers

    @pytest.fixture
    def empty_data(self):
        """Create empty test data."""
        return [], ["attr_a"], []

    @pytest.fixture
    def single_attribute_data(self):
        """Create test data with single attribute."""
        tensors = [
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
        ]
        group_identifiers = ["group1", "group2"]
        attribute_names = ["single_attr"]
        return tensors, attribute_names, group_identifiers

    @pytest.fixture
    def mixed_dtype_data(self):
        """Create test data with different tensor dtypes."""
        tensors = [
            torch.tensor([1.0, 2.0], dtype=torch.float64),  # float64
            torch.tensor([3.0, 4.0], dtype=torch.float32),  # float32
        ]
        group_identifiers = ["group1", "group2"]
        attribute_names = ["attr_a", "attr_b"]
        return tensors, attribute_names, group_identifiers

    def test_basic_construction(self, basic_data):
        """Test basic construction with valid data."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        assert len(collection) == 2
        assert collection.get_n_attributes() == 3
        assert collection.get_n_groups() == 2
        assert "group1" in collection
        assert "group2" in collection
        assert "group3" not in collection

    def test_construction_without_validation(self, basic_data):
        """Test construction with validation disabled."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers, validate=False)

        assert len(collection) == 2
        assert collection.get_n_attributes() == 3

    def test_construction_with_nan_values(self, basic_data):
        """Test construction fails with NaN values when validation enabled."""
        tensors, attribute_names, group_identifiers = basic_data
        tensors[0][0] = float("nan")

        with pytest.raises(ValueError, match="Group 'group1' contains 1 NaN values in attributes: \\['attr_a'\\]"):
            StaticAttributeCollection(tensors, attribute_names, group_identifiers)

    def test_construction_with_mismatched_attributes(self, basic_data):
        """Test construction fails with mismatched attribute count."""
        tensors, attribute_names, group_identifiers = basic_data
        tensors[0] = torch.tensor([1.0, 2.0], dtype=torch.float32)  # 2 attributes instead of 3

        with pytest.raises(ValueError, match="Group 'group1' has 2 attributes, expected 3"):
            StaticAttributeCollection(tensors, attribute_names, group_identifiers)

    def test_construction_with_wrong_dimensions(self, basic_data):
        """Test construction fails with non-1D tensors."""
        tensors, attribute_names, group_identifiers = basic_data
        tensors[0] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)  # 2D tensor

        with pytest.raises(ValueError, match="Group 'group1' tensor has 2 dimensions, expected 1"):
            StaticAttributeCollection(tensors, attribute_names, group_identifiers)

    def test_get_group_attributes_valid(self, basic_data):
        """Test get_group_attributes with valid inputs."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        result = collection.get_group_attributes("group1")
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

        result = collection.get_group_attributes("group2")
        expected = torch.tensor([4.0, 5.0, 6.0])
        assert torch.equal(result, expected)

    def test_get_group_attributes_invalid_group(self, basic_data):
        """Test get_group_attributes with invalid group."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        with pytest.raises(KeyError, match="Group 'invalid' not found in collection"):
            collection.get_group_attributes("invalid")

    def test_get_group_attribute_valid(self, basic_data):
        """Test get_group_attribute with valid inputs."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        result = collection.get_group_attribute("group1", "attr_a")
        expected = torch.tensor(1.0)
        assert torch.equal(result, expected)

        result = collection.get_group_attribute("group2", "attr_c")
        expected = torch.tensor(6.0)
        assert torch.equal(result, expected)

    def test_get_group_attribute_invalid_attribute(self, basic_data):
        """Test get_group_attribute with invalid attribute."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        with pytest.raises(
            KeyError, match="Attribute 'invalid' not found. Available: \\['attr_a', 'attr_b', 'attr_c'\\]"
        ):
            collection.get_group_attribute("group1", "invalid")

    def test_get_group_attribute_invalid_group(self, basic_data):
        """Test get_group_attribute with invalid group."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        with pytest.raises(KeyError, match="Group 'invalid' not found in collection"):
            collection.get_group_attribute("invalid", "attr_a")

    def test_properties_immutability(self, basic_data):
        """Test that properties return immutable copies."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        # Test group_identifiers
        groups = collection.group_identifiers
        original_groups = collection.group_identifiers
        groups.append("new_group")
        assert collection.group_identifiers == original_groups

        # Test attribute_names
        attributes = collection.attribute_names
        original_attributes = collection.attribute_names
        attributes.append("new_attr")
        assert collection.attribute_names == original_attributes

        # Test attribute_indices
        indices = collection.attribute_indices
        original_indices = collection.attribute_indices
        indices["new_attr"] = 99
        assert collection.attribute_indices == original_indices

    def test_attribute_indices_property(self, basic_data):
        """Test attribute_indices property."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        indices = collection.attribute_indices
        assert indices["attr_a"] == 0
        assert indices["attr_b"] == 1
        assert indices["attr_c"] == 2

        # Test immutability
        indices["new_attr"] = 3
        assert "new_attr" not in collection.attribute_indices

    def test_summary_with_data(self, basic_data):
        """Test summary method with data."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        summary = collection.summary()

        assert summary["n_groups"] == 2
        assert summary["n_attributes"] == 3
        assert summary["memory_mb"] >= 0  # Memory should be non-negative
        assert summary["attribute_names"] == ["attr_a", "attr_b", "attr_c"]
        assert summary["has_missing_values"] is False

    def test_summary_empty(self, empty_data):
        """Test summary method with empty data."""
        tensors, attribute_names, group_identifiers = empty_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        summary = collection.summary()

        assert summary["n_groups"] == 0
        assert summary["n_attributes"] == 1
        assert summary["memory_mb"] == 0.0
        assert summary["attribute_names"] == ["attr_a"]
        assert summary["has_missing_values"] is False

    def test_summary_with_missing_values(self, basic_data):
        """Test summary method detects missing values."""
        tensors, attribute_names, group_identifiers = basic_data
        tensors[0][0] = float("nan")
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers, validate=False)

        summary = collection.summary()
        assert summary["has_missing_values"] is True

    def test_repr_string(self, basic_data):
        """Test __repr__ method."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        repr_str = repr(collection)
        assert "StaticAttributeCollection" in repr_str
        assert "n_groups=2" in repr_str
        assert "n_attributes=3" in repr_str
        assert "memory_mb=" in repr_str

    def test_special_methods(self, basic_data):
        """Test special methods (__len__, __contains__)."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        # Test __len__
        assert len(collection) == 2

        # Test __contains__
        assert "group1" in collection
        assert "group2" in collection
        assert "group3" not in collection

    def test_single_attribute_collection(self, single_attribute_data):
        """Test collection with single attribute."""
        tensors, attribute_names, group_identifiers = single_attribute_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        assert collection.get_n_attributes() == 1
        assert len(collection.attribute_names) == 1
        assert collection.attribute_names[0] == "single_attr"

        # Test attribute retrieval
        result = collection.get_group_attribute("group1", "single_attr")
        expected = torch.tensor(1.0)
        assert torch.equal(result, expected)

    def test_memory_calculation_different_dtypes(self, mixed_dtype_data):
        """Test memory calculation with different tensor dtypes."""
        tensors, attribute_names, group_identifiers = mixed_dtype_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        summary = collection.summary()
        # Memory should be non-negative and account for different dtypes
        assert summary["memory_mb"] >= 0

    def test_validation_logging_success(self, basic_data, caplog):
        """Test that validation logs success message."""
        tensors, attribute_names, group_identifiers = basic_data

        with caplog.at_level(logging.INFO):
            StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        assert "Validation passed for 2 groups with 3 attributes" in caplog.text

    def test_validation_logging_empty(self, empty_data, caplog):
        """Test that validation logs warning for empty collection."""
        tensors, attribute_names, group_identifiers = empty_data

        with caplog.at_level(logging.WARNING):
            StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        assert "StaticAttributeCollection is empty" in caplog.text

    def test_edge_case_single_group(self):
        """Test collection with single group."""
        tensors = [
            torch.tensor([1.0, 2.0], dtype=torch.float32),
        ]
        group_identifiers = ["only_group"]
        attribute_names = ["attr_a", "attr_b"]

        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        assert collection.get_n_groups() == 1
        assert len(collection) == 1
        assert "only_group" in collection

        # Test data retrieval
        result = collection.get_group_attributes("only_group")
        expected = torch.tensor([1.0, 2.0])
        assert torch.equal(result, expected)

    def test_large_number_of_attributes(self):
        """Test with larger number of attributes."""
        n_attrs = 100
        tensors = [
            torch.arange(n_attrs, dtype=torch.float32),
            torch.arange(n_attrs, n_attrs * 2, dtype=torch.float32),
        ]
        group_identifiers = ["group1", "group2"]
        attribute_names = [f"attr_{i}" for i in range(n_attrs)]

        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        assert collection.get_n_attributes() == n_attrs
        assert collection.get_n_groups() == 2

        # Test specific attribute access
        result = collection.get_group_attribute("group1", "attr_50")
        expected = torch.tensor(50.0)
        assert torch.equal(result, expected)

    def test_multiple_nan_attributes(self):
        """Test validation with multiple NaN values in different attributes."""
        tensors = [
            torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float32),
        ]
        group_identifiers = ["group1"]
        attribute_names = ["attr_a", "attr_b", "attr_c"]
        tensors[0][2] = float("nan")  # Set attr_c to NaN as well

        with pytest.raises(
            ValueError, match="Group 'group1' contains 2 NaN values in attributes: \\['attr_b', 'attr_c'\\]"
        ):
            StaticAttributeCollection(tensors, attribute_names, group_identifiers)

    def test_tensor_dtype_preservation(self, mixed_dtype_data):
        """Test that tensor dtypes are preserved."""
        tensors, attribute_names, group_identifiers = mixed_dtype_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        # Original tensors should maintain their dtypes
        group1_result = collection.get_group_attributes("group1")
        group2_result = collection.get_group_attributes("group2")

        assert group1_result.dtype == torch.float64
        assert group2_result.dtype == torch.float32

    def test_sorted_group_identifiers(self):
        """Test that group identifiers are returned in the order provided."""
        tensors = [
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([3.0], dtype=torch.float32),
        ]
        group_identifiers = ["zebra", "alpha", "beta"]
        attribute_names = ["attr_a"]

        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        identifiers = collection.group_identifiers
        assert identifiers == ["zebra", "alpha", "beta"]  # Should be in provided order

    def test_integer_based_access(self, basic_data):
        """Test integer-based access methods (fast path)."""
        tensors, attribute_names, group_identifiers = basic_data
        collection = StaticAttributeCollection(tensors, attribute_names, group_identifiers)

        # Test get_group_attributes_by_idx
        result = collection.get_group_attributes_by_idx(0)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.equal(result, expected)

        result = collection.get_group_attributes_by_idx(1)
        expected = torch.tensor([4.0, 5.0, 6.0])
        assert torch.equal(result, expected)

        # Test get_group_attribute_by_idx
        result = collection.get_group_attribute_by_idx(0, "attr_a")
        expected = torch.tensor(1.0)
        assert torch.equal(result, expected)

        # Test out of bounds
        with pytest.raises(IndexError, match="Group index 2 out of bounds"):
            collection.get_group_attributes_by_idx(2)

        # Test group_to_idx mapping
        assert collection.group_to_idx["group1"] == 0
        assert collection.group_to_idx["group2"] == 1
