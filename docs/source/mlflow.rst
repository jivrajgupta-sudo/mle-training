MLflow
======

The workflow records its parameters, metrics, and artifacts in MLflow. Run the complete workflow with nested child runs using::

    python scripts/run_workflow.py

The default local store is the SQLite database ``mlflow.db``. Launch the tracking UI with::

    mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

The parent ``housing-workflow`` run contains ``data-ingestion``, ``model-training``, and ``model-scoring`` child runs. Tracking can be redirected with ``--tracking-uri`` and ``--experiment-name``.
