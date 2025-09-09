"""Tests for TFT model implementation."""

import pytest
import torch

from transfer_learning_publication.models.tft import (
    TFTConfig,
    TemporalFusionTransformer,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    StaticCovariateEncoder,
    InterpretableMultiHeadAttention,
    GLU,
)


class TestGLU:
    """Test suite for GLU (Gated Linear Unit)."""

    def test_glu_forward(self):
        """Test GLU forward pass."""
        input_size = 32
        batch_size = 4
        seq_len = 10

        glu = GLU(input_size)
        x = torch.randn(batch_size, seq_len, input_size)
        
        output = glu(x)
        
        assert output.shape == (batch_size, seq_len, input_size)
        # Check that output is different from input (transformation applied)
        assert not torch.allclose(output, x)


class TestGatedResidualNetwork:
    """Test suite for GatedResidualNetwork."""

    def test_grn_with_residual(self):
        """Test GRN with residual connection."""
        input_size = 32
        hidden_size = 64
        output_size = 32
        batch_size = 4
        seq_len = 10

        grn = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=0.1,
            residual=True,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        output = grn(x)

        assert output.shape == (batch_size, seq_len, output_size)

    def test_grn_without_residual(self):
        """Test GRN without residual connection."""
        grn = GatedResidualNetwork(
            input_size=32,
            hidden_size=64,
            output_size=16,
            dropout=0.1,
            residual=False,
        )

        x = torch.randn(4, 10, 32)
        output = grn(x)

        assert output.shape == (4, 10, 16)

    def test_grn_with_context(self):
        """Test GRN with context vector."""
        grn = GatedResidualNetwork(
            input_size=32,
            hidden_size=64,
            output_size=32,
            dropout=0.1,
            context_size=16,
        )

        x = torch.randn(4, 10, 32)
        context = torch.randn(4, 10, 16)
        output = grn(x, context)

        assert output.shape == (4, 10, 32)


class TestVariableSelectionNetwork:
    """Test suite for VariableSelectionNetwork."""

    def test_vsn_forward(self):
        """Test VSN forward pass."""
        vsn = VariableSelectionNetwork(
            input_dim=32,
            num_inputs=3,
            hidden_size=64,
            dropout=0.1,
        )

        # Create flattened embedding input
        batch_size = 4
        seq_len = 10
        flattened_embedding = torch.randn(batch_size, seq_len, 3 * 32)  # num_inputs * input_dim
        
        output = vsn(flattened_embedding)
        
        # VSN returns a tuple of (processed_embeddings, sparse_weights)
        assert isinstance(output, tuple)
        processed_embeddings, sparse_weights = output
        
        assert processed_embeddings.shape == (batch_size, seq_len, 64)  # hidden_size
        assert sparse_weights.shape == (batch_size, seq_len, 3)  # num_inputs
        # Check weights sum to approximately 1
        assert torch.allclose(sparse_weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)


class TestStaticCovariateEncoder:
    """Test suite for StaticCovariateEncoder."""

    def test_static_encoder(self):
        """Test static covariate encoder."""
        encoder = StaticCovariateEncoder(
            hidden_size=32,
            static_input_size=16,
            dropout=0.1,
        )

        static_input = torch.randn(8, 16)  # batch_size=8, static_size=16
        contexts = encoder(static_input)

        assert isinstance(contexts, dict)
        # The actual number of context vectors is determined internally
        for key, context in contexts.items():
            assert context.shape == (8, 32)


class TestInterpretableMultiHeadAttention:
    """Test suite for InterpretableMultiHeadAttention."""

    def test_attention_forward(self):
        """Test attention forward pass."""
        hidden_size = 64
        num_heads = 4
        batch_size = 4
        seq_len = 10

        attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
        )

        q = torch.randn(batch_size, seq_len, hidden_size)
        k = torch.randn(batch_size, seq_len, hidden_size)
        v = torch.randn(batch_size, seq_len, hidden_size)

        output, attn_weights = attention(q, k, v)

        assert output.shape == (batch_size, seq_len, hidden_size)
        # Attention weights shape might be averaged across heads
        assert attn_weights.shape[0] == batch_size
        assert attn_weights.shape[-2:] == (seq_len, seq_len)


class TestTemporalFusionTransformer:
    """Test suite for the main TFT model."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic TFT config for testing."""
        return TFTConfig(
            input_len=30,
            output_len=7,
            input_size=5,
            static_size=10,
            hidden_size=32,
            lstm_layers=1,
            num_attention_heads=2,
            dropout=0.1,
        )

    def test_model_initialization(self, basic_config):
        """Test model initialization."""
        model = TemporalFusionTransformer(basic_config)

        assert model.input_size == 5
        assert model.output_len == 7
        assert model.static_size == 10
        assert model.hidden_size == 32

    def test_forward_basic(self, basic_config):
        """Test basic forward pass."""
        model = TemporalFusionTransformer(basic_config)
        
        batch_size = 4
        x = torch.randn(batch_size, basic_config.input_len, basic_config.input_size)
        static = torch.randn(batch_size, basic_config.static_size)
        future = torch.randn(batch_size, basic_config.output_len, basic_config.future_input_size)

        output = model(x, static, future)

        assert output.shape == (batch_size, basic_config.output_len, 1)

    def test_forward_without_static(self):
        """Test forward pass without static features."""
        config = TFTConfig(
            input_len=20,
            output_len=5,
            input_size=3,
            static_size=0,  # No static features
            hidden_size=16,
        )
        model = TemporalFusionTransformer(config)

        batch_size = 4
        x = torch.randn(batch_size, config.input_len, config.input_size)
        future = torch.randn(batch_size, config.output_len, config.future_input_size)

        output = model(x, None, future)

        assert output.shape == (batch_size, config.output_len, 1)

    def test_forward_without_future(self):
        """Test forward pass without future features."""
        config = TFTConfig(
            input_len=20,
            output_len=5,
            input_size=3,
            future_input_size=0,  # No future features
            hidden_size=16,
        )
        model = TemporalFusionTransformer(config)

        batch_size = 4
        x = torch.randn(batch_size, config.input_len, config.input_size)
        static = torch.randn(batch_size, config.static_size)

        output = model(x, static, None)

        assert output.shape == (batch_size, config.output_len, 1)

    def test_forward_minimal(self):
        """Test forward pass with minimal features (only target)."""
        config = TFTConfig(
            input_len=15,
            output_len=3,
            input_size=1,  # Only target variable
            static_size=0,
            future_input_size=0,
            hidden_size=8,
        )
        model = TemporalFusionTransformer(config)

        batch_size = 2
        x = torch.randn(batch_size, config.input_len, config.input_size)

        output = model(x, None, None)

        assert output.shape == (batch_size, config.output_len, 1)

    def test_forward_with_all_features(self):
        """Test forward pass with all feature types."""
        config = TFTConfig(
            input_len=30,
            output_len=7,
            input_size=8,
            static_size=12,
            future_input_size=6,
            hidden_size=64,
            lstm_layers=2,
            num_attention_heads=4,
        )
        model = TemporalFusionTransformer(config)

        batch_size = 8
        x = torch.randn(batch_size, config.input_len, config.input_size)
        static = torch.randn(batch_size, config.static_size)
        future = torch.randn(batch_size, config.output_len, config.future_input_size)

        output = model(x, static, future)

        assert output.shape == (batch_size, config.output_len, 1)

    def test_model_training_mode(self, basic_config):
        """Test model in training mode."""
        model = TemporalFusionTransformer(basic_config)
        model.train()

        batch_size = 4
        x = torch.randn(batch_size, basic_config.input_len, basic_config.input_size)
        static = torch.randn(batch_size, basic_config.static_size)
        future = torch.randn(batch_size, basic_config.output_len, basic_config.future_input_size)

        output = model(x, static, future)

        assert output.shape == (batch_size, basic_config.output_len, 1)
        # Output should require grad in training mode
        assert output.requires_grad

    def test_model_eval_mode(self, basic_config):
        """Test model in evaluation mode."""
        model = TemporalFusionTransformer(basic_config)
        model.eval()

        batch_size = 4
        x = torch.randn(batch_size, basic_config.input_len, basic_config.input_size)
        static = torch.randn(batch_size, basic_config.static_size)
        future = torch.randn(batch_size, basic_config.output_len, basic_config.future_input_size)

        with torch.no_grad():
            output = model(x, static, future)

        assert output.shape == (batch_size, basic_config.output_len, 1)

    def test_different_batch_sizes(self, basic_config):
        """Test model with different batch sizes."""
        model = TemporalFusionTransformer(basic_config)

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, basic_config.input_len, basic_config.input_size)
            static = torch.randn(batch_size, basic_config.static_size)
            future = torch.randn(batch_size, basic_config.output_len, basic_config.future_input_size)

            output = model(x, static, future)

            assert output.shape == (batch_size, basic_config.output_len, 1)