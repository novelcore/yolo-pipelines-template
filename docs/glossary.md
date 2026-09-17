# Word list

Plain meanings for the words in this guide. One sentence each. Come back any time.

**Argo Workflows.** The tool that runs your pipeline on real machines and shows each step. Your contact gives you the link to yours.

**Artifact.** A file a run produces, such as the trained model or an example output.

**Data ref.** The name of the dataset version you train on. Usually `main`. You type it into the form.

**Dataset.** Your images and their labels, in the YOLO shape, stored in lakeFS.

**Epoch.** One full pass over your dataset during training. More epochs means more training.

**INT8.** A smaller, faster form of a model. The quantization steps produce it.

**lakeFS.** The storage where your dataset lives, and where the pipeline reads it from.

**Metric.** A number that says how good a model is, shown as a chart in MLflow.

**MLflow.** The results page. It stores each run's numbers, settings, and files.

**Model.** The thing you are training. You pick its size on the form, from `yolov8n` (small and fast) up to `yolov11x` (large and accurate).

**Parameter.** A single setting a run used. Every field you set on the form is saved as a parameter.

**Pipeline.** The whole set of steps that turns your dataset into a trained, saved model.

**Promote.** To mark a saved model version as ready for a stage, like Staging or Production.

**PTQ.** One way to make the smaller INT8 model, done after training. You pick it with the quantization mode.

**QAT.** Another way to make the smaller INT8 model, which fine tunes during training for better accuracy. You pick it with the quantization mode.

**Quantization.** The optional work of making a smaller, faster INT8 model. Off by default.

**Register.** To save a trained model with a name and a version so it can be found and promoted.

**Run.** One go of the pipeline, from Submit to a saved model.

**Step.** One job in the pipeline, like loading data or training the model.

**Submit.** The button on the run form that starts a run.
