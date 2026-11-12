# Word list

Plain meanings for the words in this guide. One sentence each. Come back any time.

**Artifact.** A file a run produces, such as the trained model or an example output.

**Dataset.** Your images and their labels, in the yolo layout, stored in lakeFS as the repository `dataset-<name>`. Its versions are tags (`v1`, `v2`, ...); you pick one in the form's `dataset` field, for example `dataset-spacecraft@v1`.

**kaosctl.** The command-line tool that comes with kaos-lib (0.5 or newer): create, push, validate, release and pull datasets, and replay a run's data.

**Epoch.** One full pass over your dataset during training. More epochs means more training.

**INT8.** A smaller, faster form of a model. The quantization steps produce it.

**lakeFS.** Your project's data store: your datasets, and the `outputs-<pipeline>` repository where runs write their weights.

**Metric.** A number that says how good a model is, shown as a chart in MLflow.

**MLflow.** The results page. It stores each run's numbers, settings, and files.

**Model.** The thing you are training. You pick its size on the form, from `yolov8n` (small and fast) up to `yolov11x` (large and accurate).

**Parameter.** A single setting a run used. Every field you set on the form is saved as a parameter.

**Pipeline.** The whole set of steps that turns your dataset into a trained, saved model.

**Promote.** To give a saved model version a label, `champion` or `challenger`, so everyone knows which one to use.

**PTQ.** One way to make the smaller INT8 model, done after training. You pick it with the quantization mode.

**QAT.** Another way to make the smaller INT8 model, which fine tunes during training for better accuracy. You pick it with the quantization mode.

**Quantization.** The optional work of making a smaller, faster INT8 model. Off by default.

**Register.** To save a trained model with a name and a version so it can be found and promoted.

**Run.** One go of the pipeline, from Submit to a saved model.

**Step.** One job in the pipeline, like loading data or training the model.

**Submit.** The button on the run form that starts a run.
