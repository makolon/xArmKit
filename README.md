# xArmKit

## TODO

- [ ] **Run pi0 with collected data using openpi and evaluate**
  - Use the `openpi` framework to run pi0 model with the collected dataset. Perform evaluation and analyze the results. This involves setting up the openpi environment, loading the collected data, running inference with pi0, and documenting the evaluation metrics.

- [ ] **Create dataset with recorded scene graphs**
  - Build a dataset that includes scene graph recordings. This requires implementing scene graph extraction during data collection, storing the graph structures alongside the robot data, and ensuring proper data format compatibility.

- [ ] **Verify PDDLStream using ViLaIn**
  - Integrate `ViLaIn` approach to validate and verify the `PDDLStream` task and motion planning pipeline. This involves connecting vision-language models to the PDDL planning system, testing planning scenarios, and verifying the correctness of generated plans.

- [ ] **Create Digital Twin with real2sim-eval and evaluate VLA**
  - Use the `real2sim-eval` framework to construct a Digital Twin environment. Evaluate Vision-Language-Action (VLA) model performance in this simulated twin environment. This includes setting up the simulation environment, transferring real-world data to sim, running VLA policies, and comparing sim vs real performance metrics.