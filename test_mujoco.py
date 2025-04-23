import mujoco as mj
import mujoco.viewer
import numpy as np

# Load the model
XML_MODEL = "<mujoco model='simple'><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>"
model = mj.MjModel.from_xml_string(XML_MODEL)
data = mj.MjData(model)

# Use the more stable viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(10000000):
        mj.mj_step(model, data)
        viewer.sync()