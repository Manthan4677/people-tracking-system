# People Tracking System

## Running Instructions

> [!IMPORTANT]
> Python version 3.11 is required

```sh
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (for linux/mac)
pip install -r requirements.txt

# Install dependencies (for windows)
pip install -r requirements.windows.txt

# Download our pretrained model
python download_model.

# Setting up environment variable for Linux
cp .env.example .env

# Setting up environment variable for windows
copy .env.example .env
```

## Model

You can get the pretrained model from [https://drive.google.com/file/d/15bs1HHfbVnimIWqy7wuGuZ7aJD6kUKZ6/view?usp=sharing](https://drive.google.com/file/d/15bs1HHfbVnimIWqy7wuGuZ7aJD6kUKZ6/view?usp=sharing)

## Aiven Account

You need to create a aiven account to use
PostgreSQL service. Tutorial can be found [here](https://www.youtube.com/watch?v=-ph7SiF0XQw).

## Setting up Environment Variable

```sh
# Setting up environment variable for Linux
cp .env.example .env
# Setting up environment variable for windows
copy .env.example .env
```
- Copy the <strong>Service URI</strong> from your Aiven Dashboard.
- Set POSTGRES_URI = "<strong>Service URI</strong>" in .env file
