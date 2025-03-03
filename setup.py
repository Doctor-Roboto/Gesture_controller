from setuptools import setup, find_packages

setup(
    name="gesture_controller",
    version="0.1.0",
    description="A package for controlling robots using hand gestures.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Doctor-Roboto/Gesture_controller.git",
    author="Luke Sanchez",
    author_email="sanchluk@oregonstate.edu",  
    license="Apache-2.0",
    packages=find_packages(include=["gesture", "gesture.*"]),
    install_requires=[
        "numpy>=1.24.4",
        "mediapipe>=0.10.18",
        "opencv-python>=4.10.0.84",
        "matplotlib>=3.7.2"
    ],
    python_requires=">=3.11.5",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
