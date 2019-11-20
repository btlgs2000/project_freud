
# neoEYED Keystroke analysis


## Introduction
Keystroke analysis, the way users type information on a keyboard, has become very important in the cybersecurity space to enforce strong authentication and detect impersonation frauds. 
The industry is looking for solutions with the ability to recognize users with sufficient accuracy just by looking at 3-5 times how the users types their email/passwords.


## Data collection methodology
We created a mobile application for android devices that collects keystroke information of the users by asking them to type a randomly generated sentence (random words with letters and space for a total of 30-50 characters). 
Once a user types the sentence for the first time, the sentence becomes his "passphrase" and he will have to type the same for more than 50 times on the same device. To increase variance on the keystroke behavior, the user can consecutively type the passphrase only for 3 times, then he will have to rest for 30 seconds before continuing with the data collection. 
This batch of users will be called "legit batch".
Upon data has been collected, we distributed a second mobile application, still for android devices, where all previously typed sentences have been stored. A new batch of users, called "fraudester batch", have been asked to type each sentence 3 times in a row. 
There is a slightly chance that some user of the fraudster batch is also be part of the legit batch, but on a very rare occurrence.
	

## Dataset information
### Metadata
- "user_id" is the column that identifies the account with a unique passphrase. Samples of a user includes both fraudulent and legit samples all together
- "expected_result" identifies if the sample has been typed by the legit user (marked as "1") or the fraudster (marked as "-1")
- "index" a unique identificator of the sample
- "timestamp" a time indicator when the sample was collected (not to be user for training/testing as it would bias the prediction)

### Features
- "inputs_ticks_td": Time series when the keys were pressed (key-down). 0 correspond to when the time the first key was pressed
- "inputs_ticks_tu": Time series when the keys were released (key-up). You can merge this series with 
- "inputs_ticks_ud": Time intervals between a key-up and a subsequent key-down event
- "inputs_ticks_du": Time intervals between a key-down and a subsequent key-up event
- "inputs_ticks_uu": Time intervals between a key-up and a subsequent key-up event
- "inputs_ticks_du": Time intervals between a key-down and a subsequent key-down event


## Goal
Is necessary to create a model with the ability to distinguish between the fraudulent and the legit sample of a user at the following conditions: 
- using at max 5 legitimate samples coming from the same user as training sample (if required)
- test at least 10 legitimate and 10 fraudulent samples for each user and for at least 10 users 

Expected performances: 
- False Positive Ratio (failed fraudster detection): <5%
- False Negative Ratio (failed legitimate detection): <15%
	

## Restriction and limitation
- When testing the dataset is important to balance all users and legit/fraudster data. Hence, each "userset" (all samples that belong to a single user) must have same fixed amount of samples and, within the userset, legit samples and fraudster ones must be balanced and having the exact same amount. 
- Use of fraudster data for training is only allowed if you are training a general AI for all new users (coming from a different set used from training). You cannot train an AI with both positive and negative samples of a user and then test with the remaining samples as in reality you don't have the fraudster samples. 
Example of proper performance testing: 

Testing using 3 training samples, 5 ligit samples, 5 fraud samples
User                                                           	P	N	TP	TN	FP	FN	UP	FPR		FNR		UPR		ACC		F1S
6b30996c-8fe6-4854-af5a-cc1d670f	      	5	5	5	5	0	0	0	0.00%	0.00%	0.00%	100.00%	100.00%
7b310034-e7af-42b8-8988-95798ddf       	5	5	3	4	0	1	0	12.00%	32.00%	0.00%	78.00%	72.79%
3a11cb5d-1493-4428-90ea-92971673      	5	5	3	4	0	2	0	8.00%	40.00%	0.00%	76.00%	70.18%
d58e169c-943c-42f8-b061-7b882e38       	5	5	4	4	0	1	0	8.00%	20.00%	0.00%	86.00%	81.67%
c4a5b94a-baa5-4bc9-afec-5631c80d       	5	5	3	4	0	1	0	12.00%	24.00%	0.00%	82.00%	78.94%
ec00c78d-8221-4fcf-bdfd-1dcfa69d         	5	5	3	3	2	1	0	40.00%	24.00%	0.00%	68.00%	68.83%
