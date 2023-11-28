<h1 align="center">Trading AI using TensorFlow</h1>

<p>Welcome to my repository dedicated to explain the construction of a Trading code using Deep-Learning!

</p>



<br>

<h2 align="center">🌅 Journey Highlights 🌅</h2>
<p>
After writing codes for Algorithmic Trading <a href="https://github.com/trystan-geoffre/Algorithmic-Trading">(Project Link)</a> and Deep Learning codes using TensorFlow <a href="https://github.com/trystan-geoffre/Deep-Learning-TensorFlow">(Project Link)</a> I wanted to combine both in order to build a powerfull tool. By merging these two disciplines, I aim to create a synergistic tool that can leverage historical market data, recognize complex patterns, and adapt dynamically to evolving market conditions. This fusion allows for a more robust and adaptive trading system, where the algorithms can learn from data patterns, optimize decision-making processes, and potentially uncover hidden opportunities that may be challenging for traditional strategies to discern.
  

<h2 align ="center">🎯 Model review 🎯 </h2>


 The model has been trained on historical S&P 500 data, which was normalized to mitigate skewness. The model uses Bidirectional LSTM (Long Short Term Memory) layers. Leveraging Bidirectional LSTM (Long Short Term Memory) layers proves advantageous for Time Series forecasting. These layers possess the unique capability to capture context in both the forward and backward directions, significantly enhancing pattern recognition. Furthermore, the model incorporates essential components such as a learning rate scheduler, checkpoints, and an early stopping mechanism. Checkpoints permit to save the best-performing model based on validation loss. Early stopping is implemented to prevent overfitting, optimizing the model's generalization ability. The learning rate scheduler dynamically fine-tunes the learning rate throughout the training process, contributing to the model's adaptability and performance refinement.

---


<h2 align="center">🔎 Repository Overview 🔍</h2>

In the development of our Trading AI, we follow a structured process comprising four essential steps:

<br>

<details>
  <h2 align="center">📜 Get Data from S&P 500 📜</h2>
  
  <summary> 📜 Get Data from S&P 500 📜</summary> 

  <p>
First you must create a file called "config.py" where you will reiseign API_KEY, SECRET_KEY , OAUTH_TOKEN for Alpaca API.

Then use "Get_DataS&P.py". The code utilizes the yfinance library to download historical stock price data for companies listed in the S&P 500 index. The list of S&P 500 tickers is obtained from Wikipedia. The script defines a date range for the last 5 years and creates a directory to store the downloaded data. It then iterates through each S&P 500 company, downloads its historical data using yfinance, and saves it as a CSV file in the specified directory. <a href="https://github.com/trystan-geoffre/Trading-AI-TensorFlow/blob/master/Get_Data-S%26P.py"> Code Link</a>
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center">♟️ Model Strategy ♟️</h2>
  
  <summary> ♟️ Model Strategy ♟️</summary> 

  <p>
The Python script "Model_Strategy.py" is designed to construct a deep learning model for time series forecasting using historical stock price data from S&P 500 companies. Helper functions are defined for data normalization and creating windowed datasets. The main model function reads historical stock data, normalizes it, and splits it into training and validation sets. Then we define a Bidirectional LSTM model with multiple layers and compil it. The model is trained using windowed datasets, incorporating callbacks such as learning rate scheduling, early stopping, and model checkpointing. The trained model is saved as "mymodel.h5" for future use. Overall, the script showcases the construction of an effective deep learning model for time series forecasting, utilizing advanced features and callbacks for enhanced performance.<a href="https://github.com/trystan-geoffre/Trading-AI-TensorFlow/blob/master/Model_Strategy.py"> Code Link</a>
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center">🕸️ Testing the Model 🕸️</h2>
  
  <summary> 🕸️ Testing the Model 🕸️</summary> 

  <p>
In the "Test_Model" script, we initiate the process by downloading time series data, performing normalization, and dividing it into distinct training and testing sets. Following this, the pre-trained deep learning model, we saved as "mymodel.h5," is loaded. Subsequently, the model is employed to generate predictions on the test dataset, and these predictions are then denormalized. The evaluation of model performance is conducted through the computation of the Mean Absolute Error (MAE) using NumPy arrays, providing insights into the accuracy of the predictions in comparison to the actual test data. <a href="https://github.com/trystan-geoffre/Trading-AI-TensorFlow/blob/master/Test_Model.py"> Code Link</a>
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> ⚡️ Lunching the model to real-time Traiding< ⚡️/h2>
  
  <summary> ⚡️ Lunching the model to real-time Traiding< ⚡️</summary> 
<p>
The code "Live_Trading.py" begins by loading the pre-trained deep learning model ("mymodel.h5") to make predictions. The trading strategy is based on these predictions, with specific conditions triggering buy and sell orders. Additionally, the script includes backtesting functionality, connecting to the Alpaca API for live trading, and a schedule for running the trading strategy at regular intervals. The main execution section initiates live trading with predefined trading pairs like AAPL, SPY, MSFT, and META. The script is designed to operate continuously, running the trading strategy based on the specified time frame. <a href="https://github.com/trystan-geoffre/Trading-AI-TensorFlow/blob/master/Live_Trading.py"> Code Link</a>
  </p>
  <br>
</details>

<br>

This is the end of this depository on Trading AI using TensorFlow. If you want to see more complexe projects on Trading I would invite invite you to see <a href="https://github.com/trystan-geoffre/Full-Stack-Trading-App">Full Stack Trading App</a>

  
