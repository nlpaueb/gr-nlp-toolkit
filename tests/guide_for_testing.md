In Visual Studio Code, make sure:
- you have created a virtual environment (venv) with the corresponding python version supported
- you have installed the venv dependencies `pip install -r requirements.txt`
- Configure this interpreter (`ctrl + shift + p` -> `Python: Select Interpreter`)
- **Important**: Install the toolkit as a `package` in *editable* mode via `pip install -e .` (run this command from the root directory of the project!)
- Configure the tests accordingly (`ctrl + shift + p`). Select `unittest` as your test framework, `tests` as the directory containing the tests and `test_*` as the file patterns to be matched as test files. This shall create a `.vscode/settings.json` file like this for you:
```json
{
    "python.testing.unittestArgs": [
        "-v",
        "-s",
        "tests",
        "-p",
        "test_*.py"
    ],
    "python.testing.unittestEnabled": true,
}
```
- Now, go to the `Testing` tab in the left column in Visual Studio Code and click the `Refresh Tests` button in order to discover them.


<br>Bonus (for Windows):
<br>
If you use Windows, it would be helpful if you enable the developer mode. This will speed things up in the caching mechanism under-the-hood of huggingface hub. (https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)