# README

This folder contains a `.env` file to be used with `python-dotenv` and an `api_configs.json` file to be used with Azure endpoints.

## Use of `python-dotenv` and the `.env` file

After installing `python-dotenv` you can use the `.env` file as follows:

```python
import openai
import os
import dotenv

_ = dotenv.load_env()
openai.api_key = os.get_env('OPENAI_API_KEY')
models = openai.Model.list() # To see if everything works.
```

