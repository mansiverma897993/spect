# Spam Detection Web Application

A very simple yet effective Spam Detection Web App built with **Python**, **Streamlit**, and **Scikit-Learn**. 
It uses TF-IDF Vectorization and a Multinomial Naive Bayes algorithm to classify messages as `Spam` or `Not Spam`.

## Project Structure
* `requirements.txt`: List of python dependencies.
* `train_model.py`: Automates dataset downloading, text preprocessing, and trains the machine learning model.
* `app.py`: The beautiful frontend built with Streamlit!

## How to Run Locally

You can run this project locally on your system using the following commands:

**1. Clone/Download the repository**
```bash
# Navigate to your desired folder in terminal, then download the files.
# Make sure you are in the project folder:
cd path/to/spamdetect
```

**2. Create a Virtual Environment (Recommended)**
*(This keeps dependencies clean)*
```bash
python -m venv venv

# Activate on Windows:
venv\\Scripts\\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the AI Model**
This will download the dataset, verify there is no overfitting between Training and Testing accuracy, and save the `.pkl` file.
```bash
python train_model.py
```

**5. Start the Web App**
```bash
python -m streamlit run app.py
```
A browser window will automatically open at `http://localhost:8501`.

## Easy Deployment
To deploy this project for free (e.g., to Streamlit Community Cloud):
1. Push this folder to a GitHub repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click "New App" and select your repository, branch, and `app.py` stream.
4. Streamlit will automatically install `requirements.txt` and host your project in seconds!
