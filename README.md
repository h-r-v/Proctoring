**# Introduction:**

This project is a set of two tools created to proctor online exams.

The tools are:

- Lip Movement AI
- Volume Meter

**# Working overview**

This project contains two parts:

1. Lip Movement AI:

This tool is used to proctor the students only based on their lip movement. Any time the student opens his/her mouth the software raises a flag.

2. Volume Meter

This tool is used to proctor the students only based on their surrounding sound. Any time the surrounding noise of the student increases above a threshold a flag is raised.

**# Installation and Launching for lip movement AI**

1. Installing python 3.8.5

		- wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tgz
		- tar -xvf Python-3.8.5.tgz
		- cd Python-3.8.5
		- ./configure
		- make
		- make altinstall

2. Create virtual environment

		- python3 -m venv env\_LM

3. Activate environment and install dependencies:

		- source env\_LM/bin/activate
		- pip install -r requirements.txt

1. Copy the package to the right path:

		- Copy the &#39;&#39;lip movement AI&quot; dir to the dir where you want to use it.

4. Same use:

		- Use the link provided below to get familiar with how to use the library:

		[https://drive.google.com/file/d/1itKL0egPPlfJj9BkVD7p8EsFoKeQfzoz/view?usp=sharing](https://drive.google.com/file/d/1itKL0egPPlfJj9BkVD7p8EsFoKeQfzoz/view?usp=sharing)

**# Installation and Launching for volume meter**

1. Launching the tool:

		- Launch &#39;index.html&#39;

**# Directory Structure for lip movement AI**

- models dir: contains the pretrained facial landmark detection model
- face\_detector.py: helper file for lipmovementUtil.py. This file is used to load a face detector, find faces and make bounding boxes around them.
- landmark\_detector.py: helper file for lipmovementUtil.py. This file is used to load a landmark detector and make circles on them.
- lipmovementUtil.py: this is the file that will be accessed by the user to access the functionality of the library.

**# Demo Video for lip movement AI**

[https://drive.google.com/file/d/1Vi89sVBOEMvy-SCjqWFsHB2H18TX7DtZ/view?usp=sharing](https://drive.google.com/file/d/1Vi89sVBOEMvy-SCjqWFsHB2H18TX7DtZ/view?usp=sharing)
