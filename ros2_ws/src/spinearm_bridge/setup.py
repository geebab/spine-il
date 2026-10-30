from glob import glob

from setuptools import setup

setup(
    name="spinearm_bridge",
    version="0.1.0",
    packages=["spinearm_bridge"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/spinearm_bridge"]),
                ("share/spinearm_bridge", ["package.xml"]),
                ("share/spinearm_bridge/launch", glob("launch/*.py"))],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "arm_bridge = spinearm_bridge.arm_bridge:main",
        "tracker_bridge = spinearm_bridge.tracker_bridge:main",
        "calib_tf = spinearm_bridge.calib_tf:main",
        "description_pub = spinearm_bridge.description_pub:main",
        "recording_player = spinearm_bridge.recording_player:main",
    ]},
)
