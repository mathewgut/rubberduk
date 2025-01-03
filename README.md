# rubberduk
A EULA, TOS, and Privacy Policy scanner that identifies concerning clauses specifically relating to user privacy, data, and security.
<h2> Mission Statement </h2>
rubberduk is an attempt at making a system to allow for users of online platforms to be more informed of the conditions and terms within TOS's, EULA's, and Privacy Policies when using said platforms. The goal is to give users more power over their data and consent through knowledge and understanding of agreements and policies.
<h2> Approach </h2>
We aim to accomplish this goal through a few fundamental techniques, which when broken down, are quite simple. 
</br></br>Note: Due to extremely limited resources, the approach is tailored to cost effective, and ready to use methods and systems. As we do not have the resources (time, cost, computational power) to train models on relevant data sets. In an ideal situation we would build multiple custom models trained on legal, agreement, and data privacy text.
</br></br>
<ol> <li>Zero Shot Classification <ul> <li>This utilizes natural language processing models trained on significantly large corpuses of data, where they will then attempt to match a given set of text to a set of labels. It will then produce scores for how close it believes a specific section of text matches a set label</li></ul></li> <li>Multi Model Fusion <ul><li> We use multiple models on the same section of text and average out their combined scores. Utilizing multiple models allows for more accurate results. As different models have been trained with different data and have different biases. This will <i>hopefully</i> void the issue of biased results, as it should average out between multiple models </li></ul></li> <li>Text Batching <ul><li>To simplify the context system we have added a batching system. Which will take a documents text, cut it into chunks dependent on the specified sentence size, and pass it to the model for labelling. This is a resource efficent approach that can give the model both pre an post cursor context to <i>hopefully</i> provide more accurate estimations on label scores</li></ul></li> <li>Sliding Window<ul><li>Essentially just another context system. This uses bits of text from the previous batch in the current batch to give related context in case text is tructuated too early </li></ul></li> </ol>
<h2>Progress</h2>
This is a passion project for me which I started with the OpenAI API in November of 2022, and it has slowly evolved since then. Core systems have been developed and are working. Currently Ryan Sweetnam is helping optimize the code to help achieve the end goal. Progress is slow, as I am tied up with other priorites and projects.
<h2>End Goal</h2>
Eventually, we plan to release this as a completely free browser extension to automatically scan documents for users (similar to a grammarly system). However, there is a lot of cost involved (computational resources for modeling, servers, etc) so there is a good chance the project may stall before then.
<h2>Else</h2>
If you want to keep up with the project, or have any ideas or concerns, feel free to post an issue or contact me here on GitHub or LinkedIn.
<h3>Citations</h3>
@article{sileo2023tasksource,
  title={tasksource: Structured Dataset Preprocessing Annotations for Frictionless Extreme Multi-Task Learning and Evaluation},
  author={Sileo, Damien},
  url= {https://arxiv.org/abs/2301.05948},
  journal={arXiv preprint arXiv:2301.05948},
  year={2023}
}

cross-encoder/nli-deberta-base

License: Apache 2-0
