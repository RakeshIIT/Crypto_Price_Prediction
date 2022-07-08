# Crypto_Price_Prediction

This is the crpyto currency buy/sell predictor, where the custom CNN model is trained to predicted whether to buy or sell the crypto. 

The crypto currencies choosen here are:
1. "BCH-USD"
2. "BTC-USD"
3. "ETH-USD"
4. "LTC-USD"

The input datasets have basically six columns as shown below

<img width="415" alt="image" src="https://user-images.githubusercontent.com/64364295/178031816-36cab556-a42f-43cf-938d-b6543fd051fc.png">

After training the custom CNN model, its performance can be seen uing a tensorboard callbacks. 

After running tensorboard --logdir logs/ command CLI, the outputs will be something like this:

![MicrosoftTeams-image (1)](https://user-images.githubusercontent.com/64364295/178033158-10345c79-f096-4912-bce3-b874de016375.png)


