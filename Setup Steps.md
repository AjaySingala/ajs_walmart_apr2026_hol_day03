# Cloning the HOL Repositories
The hands-on lab (HOL) exercises will be provided in separate URLs for each day.

For e.g.; for Day 1, the url is:
https://github.com/AjaySingala/ajs_walmart_apr2026_hol_day01.git

Here are the steps to clone the repository (repo) on the VM (or your local machine) and use it:
- On the VM, open the terminal emulator.
- Enter the following command to clone the repo:
`git clone <github url>`
- Replace <github url> with the GitHub URL for the specific day.
- For e.g.; for day #1, do this:
`git clone https://github.com/AjaySingala/ajs_walmart_apr2026_hol_day01.git `
- This will create a folder named ajs_walmart_apr2026_hol_day01 in your current directory.
- Open Visual Studio Code (VSC)
- Select File -> Open Folder (or press CTRL+K+F)
- Select the folder created after cloning, for e.g.; ajs_walmart_apr2026_hol_day01 for day #1.
- Open the folder

# Environment Variables
To run your code, you will have to configure some environment variables.
Follow these steps:
- In the root folder of this project, where the file config.py exists, create the file named .env for your environment variables:
- Click on config.py
- Click on the icon to create a new file or select File -> New File from the menu
- Ener the filename as .env
- Add the following lines to this .env file and then save the file:
```python
OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
MODEL_NAME="gpt-4o-mini"
TEXT_EMBEDDING_MODEL="text-embedding-3-small"
```
- Replace <YOUR_OPENAI_API_KEY> with the Key provided to you.

## Notes:
- The .env file is to be created in the root folder of the project for the day. For e.g.; ajs_walmart_apr2026_hol_day01.
- It must be located along with the config.py file and not in any of the sub-folders.

# Setup Local Virtual Python Environment
- Open your project’s folder in VS Code (VSC).
- In the terminal window, create a new virtual environment:
`python -m venv .venv`
- If you have multiple versions of Python installed on your machine, then create the environment using the specific  python version.
	- For e.g.; if you have Python 3.10, 3.11 and 3.13 installed, but want to work with Python 3.13 for your specific project, then determine the path where the python.exe for 3.13 is installed and then use that explicit path to create the virtual environment.
	- Let’s say it is in the folder `~/Python313` or `C:\Python313`
	- Run this:
`~/Python313/python -m venv .venv`
OR
`C:\Python313\python -m venv .venv`
- Then, activate the virtual environment by running:

## On Mac/Linux:
`source .venv/bin/activate`

## On Windows:
`.venv\Scripts\Activate.ps1`

- The prompt on the terminal should show (.venv) on the left.

- Then, in VSC, press CTRL+SHIFT+P
- Search for and select “Python: Select interpreter”
- Select the python.exe from the virtual environment where it says “recommended”.
- You are all set

## Notes:
- This setting to create a virtual environment is to be done only once for the specific day’s HOLs. You don’t have to do it every time you open the folder in VS.
- If the .venv folder is deleted for whatever reason, then you will have to repeat these steps all over again.
- If CTRL+SHIFT+P does not show up the command selection dropdown at the top in VSC, do not worry.
- Your python commands will still work.

# Running Python code files
To run the python code files, do this:
- Open your project folder in VSC
- Click on View -> Terminal to open the terminal window or press CTRL+`.
- Make sure you are in the right folder of your project in the terminal window
- Every sub-folder in the project will have it’s own specific install.txt file that has the commands to install the packages required to run the code in that folder.
- Navigate to the sub-folder and open the installs.txt file.
- In the terminal windows, navigate to the folder. For e.g.;
cd demo01_tokens 
- Copy the pip install commands from the file and run them one after the other in the terminal window.
- For e.g.; for the code for day #1, in the demo01_tokens folder, the installs.txt file has these commands:
`python -m pip install openai numpy matplotlib seaborn tiktoken`
`python -m pip install python-dotenv`
- Copy the first command and paste it in the terminal window and run it.
- Then do the same for the second command.
- Execute all such pip install commands in the file.
- Once done, you can run the .py file in that folder.
- In the terminal windows, make sure you are in the right folder
- Then run the .py file. For e.g.;:
`python demo01_token_visual.py`
- It will show the output from the code execution.



