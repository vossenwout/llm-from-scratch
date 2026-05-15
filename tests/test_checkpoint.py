import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from llm_from_scratch.checkpoint import load_checkpoint, save_checkpoint
from llm_from_scratch.transformer import Transformer


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)

    vocab_size = 16
    model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=16,
        context_length=8,
        attention_heads=4,
        ff_hidden_dim=32,
        n_decoders=1,
        p_dropout=0,
    )
    optimizer = optim.Adam(model.parameters(), lr=1.0)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.1)

    X = torch.randint(0, vocab_size, (2, 8))
    y = torch.randint(0, vocab_size, (2, 8))
    loss = CrossEntropyLoss()(model(X).permute(0, 2, 1), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_dir=tmp_path,
        filename="checkpoint.pt",
        epoch=2,
        global_step=7,
        last_train_loss=1.25,
        last_val_loss=1.5,
        best_val_loss=1.4,
        model_config={"embedding_dim": 16},
        data_config={"train_dataset": "train.txt"},
        training_config={"train_epochs": 1},
        tokenizer_config={"tokenizer_type": "char"},
    )

    loaded_model = Transformer(
        vocab_size=vocab_size,
        embedding_dim=16,
        context_length=8,
        attention_heads=4,
        ff_hidden_dim=32,
        n_decoders=1,
        p_dropout=0,
    )
    loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=1.0)
    loaded_scheduler = optim.lr_scheduler.LambdaLR(
        loaded_optimizer, lr_lambda=lambda step: 0.1
    )

    epoch, global_step, best_val_loss = load_checkpoint(
        checkpoint_path=str(tmp_path / "checkpoint.pt"),
        model=loaded_model,
        optimizer=loaded_optimizer,
        scheduler=loaded_scheduler,
        device="cpu",
    )

    assert epoch == 2
    assert global_step == 7
    assert best_val_loss == 1.4
    assert loaded_optimizer.state_dict()["state"]
    assert loaded_scheduler.state_dict() == scheduler.state_dict()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded_model.state_dict()[name])
