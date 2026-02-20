"""
Fixtures para testes de integração.

Garante que o estado global do MLflow (tracking URI, _mlflow_configured,
runs ativas) seja limpo antes e depois de cada teste de integração,
evitando contaminação quando os testes rodam em conjunto.
"""
import pytest
import mlflow


@pytest.fixture(autouse=True)
def reset_mlflow_state():
    """Reseta estado global do MLflow antes e depois de cada teste.

    Problemas que isso evita ao rodar todos os testes juntos:
    1. _mlflow_configured = True de testes anteriores impede reconfiguração
    2. Tracking URI apontando para banco real (data/mlflow.db) de outro teste
    3. Run ativa "órfã" de um teste anterior que não foi fechada
    """
    import src.utils.mlflow_setup as mlflow_setup_module

    # Salvar estado original
    original_configured = mlflow_setup_module._mlflow_configured

    # Garantir estado limpo antes do teste
    mlflow_setup_module._mlflow_configured = False

    # Fechar qualquer run ativa que tenha vazado de outro teste
    if mlflow.active_run():
        mlflow.end_run()

    yield

    # Cleanup após o teste
    if mlflow.active_run():
        mlflow.end_run()

    mlflow_setup_module._mlflow_configured = original_configured
    mlflow.set_tracking_uri("")
