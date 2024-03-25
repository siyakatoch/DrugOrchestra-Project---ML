1.1 Background and Challenges

Traditional drug discovery methods often tackle one problem at a time, such as
predicting drug responses, identifying targets, or spotting side effects, without
considering how these elements are interconnected. This approach limits their
effectiveness, especially for new drugs that haven’t been extensively studied.
This is where DrugOrchestra comes into play. It combines all three tasks into
one framework, using a deep multi-task learning approach.


1.2 The Significance of DrugOrchestra

The significance of DrugOrchestra lies in its integration of these three tasks,
leveraging the interconnectedness among them to enhance prediction accuracy
for unseen drugs based solely on their molecular structures. It solves multiple
prediction tasks at the same time.


1.3 Tasks

Drug Response Prediction: This task estimates how effectively a drug will act on
a biological system, crucial for assessing its therapeutic potential and tailoring
treatments to individual genetic profiles.

Drug Target Identification: It involves identifying the molecular targets a
drug interacts with to exert its effects, key to understanding the drug’s mecha-
nism of action and optimizing its therapeutic efficacy and specificity.

Drug Side Effect Prediction: This task predicts the adverse reactions a drug
may cause, essential for evaluating drug safety and minimizing potential risks
to patients during the drug development process.


1.4 Our Code

Using different models, we have coded models to get scores to understand evaluation metrics and their performances. valuation Metrics are Area Under the Receiver Operating Characteristic Curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC) for Classification Tasks which are drug target and side effect predictions. Spearman’s Correlation Coefficient and Mean Squared Error (MSE) are used for Regression Task which is drug response prediction. The overall improved performance of MTL demonstrates the effectiveness of using multi-task learning to transfer knowledge across these three tasks.
