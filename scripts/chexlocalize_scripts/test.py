# %%
import pandas as pd

# %%
no_regularization = pd.read_csv(
    "./no_regularization/hitmiss_summary_results.csv"
)
no_regularization["AUC"] = [0.84, 0.83]
positive_regularization = pd.read_csv(
    "./positive_regularization/hitmiss_summary_results.csv"
)
positive_regularization["AUC"] = [0.84, 0.87]
negative_regularization = pd.read_csv(
    "./negative_regularization/hitmiss_summary_results.csv"
)
negative_regularization["AUC"] = [0.83, 0.87]
# %%
full_df = pd.concat(
    [no_regularization, positive_regularization, negative_regularization],
    axis=0,
)
full_df
# %%
full_df.index = (
    no_regularization.shape[0] * ["DenseNet-121"]
    + positive_regularization.shape[0] * ["DenseNet-121 with Positive regularization"]
    + negative_regularization.shape[0] * ["DenseNet-121 with Negative regularization"]
)
full_df.columns = ['Pathology', 'lower_hit_rate', 'Hit rate', 'upper_hit_rate', 'AUC']
full_df['Saliency method'] = 'GradCAM'
full_df = full_df[['Pathology', 'Saliency method', 'AUC' ,'Hit rate']]
full_df = full_df.reset_index()
full_df.columns = ['Model', *full_df.columns[1:]]
full_df

# %%
no_regularization = pd.read_csv("./no_regularization/iou_summary_results.csv")
positive_regularization = pd.read_csv(
    "./positive_regularization/iou_summary_results.csv"
)
negative_regularization = pd.read_csv(
    "./negative_regularization/iou_summary_results.csv"
)

temp_df = pd.concat(
    [no_regularization, positive_regularization, negative_regularization],
    axis=0,
)
full_df['mIoU'] = temp_df['mean'].reset_index(drop=True)
# %%
print(full_df.to_latex(index=False))
# %%
full_df.sort_values(by='Pathology')
# %%
\begin{tabular}{ll|lccc}
\toprule
                                    \textbf{Model} &                  \textbf{Pathology} & \textbf{Saliency method} &  \textbf{AUC} &  \textbf{Hit rate} &  \textbf{mIoU} \\
\midrule
                             DenseNet-121 &                Atelectasis &         GradCAM & 0.84 &     0.075 & 0.067 \\
                             DenseNet-121 & E. Card. &         GradCAM & 0.83 &     0.201 & 0.186 \\
DenseNet-121 with pos. reg. &                Atelectasis &         GradCAM & 0.84 &     0.364 & 0.167 \\
DenseNet-121 with pos. reg. & E. Card. &         GradCAM & 0.87 &     0.607 & 0.367 \\
DenseNet-121 with neg. reg. &                Atelectasis &         GradCAM & 0.83 &     0.013 & 0.065 \\
DenseNet-121 with neg. reg. & E. Card. &         GradCAM & 0.87 &     0.073 & 0.174 \\
\bottomrule
\end{tabular}