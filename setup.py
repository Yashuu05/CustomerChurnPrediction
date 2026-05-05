from setuptools import setup, find_packages

setup(
    name='churn_prediction_package',
    version='0.0.1',
    description='A Python package',
    author='Yash Chillal',
    author_email='chillalyash2005@gmail.com',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'xgboost',
        'mlflow',
        'python-dotenv',
        'pymongo',
        'lightgbm',
        'imblearn'
    ],
)