# Authentication using Keystroke Dynamics with Deep Neural Networks

Passwords have formed the backbone of day-to-day security systems, but in many aspects they are flawed. According to a [report](https://www.ciodive.com/news/an-average-95-passwords-stolen-per-second-in-2016-report-says/435204/#:~:text=Dive%20Brief%3A,95%20passwords%20stolen%20every%20second.) an average of 95 passwords are stolen per second globally. 

To augment this flawed system, I looked some other forms of security that would not require further investment in hardware. Through research, it is ascertained that the typing habits of individuals vary subtly across the population and as such can be used to discern people by the 'biometric signature' that their typing represents. With an adequate model, we could therefore determine the validity of the purported 'real' user by the manner in which they type their password.

To be explicit, I propose a model where a user types their password into the machine once. The machine will identify whether or not the user is who they say they are, with a high degree of precision and accuracy. I seek to minimize, in particular, the false-positive rate, as this would correspond to a false user being allowed into the system accidentally.

In order to train the system, the user would have to enter their password more than once.
