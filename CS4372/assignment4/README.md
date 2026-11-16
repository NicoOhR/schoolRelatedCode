The code provided for this assignment in assignment4.py was written locally and
ran on google colab's T4 free tier GPU's through the google colab cli. I believe
the code should be runnable through the google colab notebook environment but
I have not tested it for myself. The commands to run this code via the colab cli
are 

```
colab new --gpu T4
colab exec --session <session id given by the above command> --file ./assignment4.py
```

The code will run on the google colab server give back the results from the
experiments.
