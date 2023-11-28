<h1 align="center">Trading AI using TensorFlow</h1>

<br>

Welcome to my repository dedicated to explain the construction of a Trading code using Deep-Learning!

<br>

<h2 align="center">🌅 Journey Highlights 🌅</h2>
<p>
After writing codes for Algorithmic Trading <a href="https://github.com/trystan-geoffre/Algorithmic-Trading">(Project Link)</a> and Deep Learning codes using TensorFlow <a href="https://github.com/trystan-geoffre/Deep-Learning-TensorFlow">(Project Link)</a> I wanted to combine both in order to build a powerfull tool. By merging these two disciplines, I aim to create a synergistic tool that can leverage historical market data, recognize complex patterns, and adapt dynamically to evolving market conditions. This fusion allows for a more robust and adaptive trading system, where the algorithms can learn from data patterns, optimize decision-making processes, and potentially uncover hidden opportunities that may be challenging for traditional strategies to discern.
  

<h2 align ="center">🎯 Model review 🎯 </h2>


  The model has undergone training S&P 500 historical data which has been normalized in order to reduce the skewness of it. The model uses Bidirectional LSTM (Long Short Term Memory) layers. Leveraging Bidirectional LSTM (Long Short Term Memory) layers proves advantageous for Time Series forecasting. These layers possess the unique capability to capture context in both the forward and backward directions, significantly enhancing pattern recognition. Furthermore, the model incorporates essential components such as a learning rate scheduler, checkpoints, and an early stopping mechanism. Checkpoints permit to save the best-performing model based on validation loss. Early stopping is implemented to prevent overfitting, optimizing the model's generalization ability. The learning rate scheduler dynamically fine-tunes the learning rate throughout the training process, contributing to the model's adaptability and performance refinement.






  
