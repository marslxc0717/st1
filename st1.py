import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 常见的回归模型
models = {
    "线性回归": LinearRegression,
    "决策树": DecisionTreeRegressor,
    "随机森林": RandomForestRegressor,
    "梯度提升": GradientBoostingRegressor,
    "支持向量回归": SVR,
    "K 最近邻": KNeighborsRegressor,
    "神经网络 (MLP)": MLPRegressor,
    "高斯过程": GaussianProcessRegressor,
    "岭回归": Ridge,
    "Lasso 回归": Lasso,
}

st.sidebar.title("选项")

# 主选择框
option = st.sidebar.selectbox("请选择操作", ["训练模型", "预测结果", "数据分析"])

if option == "训练模型":
    st.title("训练机器学习模型")
    uploaded_file = st.file_uploader("上传数据集 (Excel 格式)", type=["xlsx", "xls"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("数据集预览：", df.head())

        features = st.multiselect("选择特征列", df.columns)
        label = st.selectbox("选择标签列", df.columns)

        if features and label:
            X = df[features]
            y = pd.to_numeric(df[label], errors='coerce')  # 转换目标列为数值类型

            if y.isna().any():
                st.warning("目标列包含无效值或缺失值，已删除对应行。")
                X = X[~y.isna()]
                y = y.dropna()

            normalize = st.radio("是否进行归一化处理", ["不归一化", "标准化 (StandardScaler)", "归一化 (MinMaxScaler)"])
            scaler = None
            if normalize == "标准化 (StandardScaler)":
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            elif normalize == "归一化 (MinMaxScaler)":
                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)

            model_name = st.selectbox("选择回归算法", list(models.keys()))
            model_class = models[model_name]
            st.subheader(f"{model_name} 超参数设置")
            params = {}
            if model_name == "线性回归":
                st.write("此算法没有可调节的超参数。")
            elif model_name == "决策树":
                params["max_depth"] = st.number_input("Maximum Depth (max_depth)", 1, 100, 10)
                params["min_samples_split"] = st.number_input("Minimum Samples Split (min_samples_split)", 2, 100, 2)
                params["min_samples_leaf"] = st.number_input("Minimum Samples Leaf (min_samples_leaf)", 1, 100, 1)
            elif model_name == "随机森林":
                params["n_estimators"] = st.number_input("Number of Trees (n_estimators)", 10, 500, 100)
                params["max_depth"] = st.number_input("Maximum Depth (max_depth)", 1, 100, 10)
                params["min_samples_split"] = st.number_input("Minimum Samples Split (min_samples_split)", 2, 100, 2)
                params["min_samples_leaf"] = st.number_input("Minimum Samples Leaf (min_samples_leaf)", 1, 100, 1)
            elif model_name == "梯度提升":
                params["n_estimators"] = st.number_input("Number of Trees (n_estimators)", 10, 500, 100)
                params["learning_rate"] = st.slider("Learning Rate (learning_rate)", 0.01, 1.0, 0.1)
                params["max_depth"] = st.number_input("Maximum Depth (max_depth)", 1, 100, 10)
                params["min_samples_split"] = st.number_input("Minimum Samples Split (min_samples_split)", 2, 100, 2)
                params["min_samples_leaf"] = st.number_input("Minimum Samples Leaf (min_samples_leaf)", 1, 100, 1)

            if st.button("训练模型"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = model_class(**params)
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                st.write(f"训练集 R²: {r2_score(y_train, train_pred):.4f}")
                st.write(f"测试集 R²: {r2_score(y_test, test_pred):.4f}")

                # 保存模型到内存并提供下载
                model_data = {"model": model, "scaler": scaler}  # 保存模型和归一化方法
                model_buffer = io.BytesIO()
                pickle.dump(model_data, model_buffer)
                model_buffer.seek(0)
                st.download_button(
                    label="下载模型",
                    data=model_buffer,
                    file_name=f"{model_name}_model.pkl",
                    mime="application/octet-stream"
                )

                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].scatter(y_train, train_pred, alpha=0.7)
                ax[0].plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
                ax[0].set_title("Training: Actual vs Predicted")
                ax[0].set_xlabel("Actual Values")
                ax[0].set_ylabel("Predicted Values")

                ax[1].scatter(y_test, test_pred, alpha=0.7, color='orange')
                ax[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
                ax[1].set_title("Testing: Actual vs Predicted")
                ax[1].set_xlabel("Actual Values")
                ax[1].set_ylabel("Predicted Values")

                st.pyplot(fig)

elif option == "预测结果":
    st.title("模型预测结果")
    model_file = st.file_uploader("上传训练好的模型 (.pkl 文件)", type=["pkl"])
    data_file = st.file_uploader("上传待预测的数据集 (Excel 格式)", type=["xlsx", "xls"])

    if model_file and data_file:
        model_data = pickle.load(model_file)
        model = model_data["model"]
        scaler = model_data["scaler"]
        df = pd.read_excel(data_file)
        st.write("待预测数据集预览：", df.head())

        predict_features = st.multiselect("选择用于预测的特征列", df.columns)
        if predict_features:
            X = df[predict_features]

            # 应用训练时相同的归一化或标准化方法
            if scaler is not None:
                X = scaler.transform(X)

            predictions = model.predict(X)
            df["Predictions"] = predictions
            st.write("预测结果：", df)

            # 下载结果
            output_buffer = io.BytesIO()
            df.to_excel(output_buffer, index=False)
            output_buffer.seek(0)
            st.download_button(
                label="下载预测结果",
                data=output_buffer,
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

elif option == "数据分析":
    st.title("数据分析工具")
    uploaded_file = st.file_uploader("上传数据集 (Excel 格式)", type=["xlsx", "xls"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("数据集预览：", df.head())
        analysis_option = st.radio("选择分析功能", ["变量聚类", "相关性分析", "灰色关联分析", "随机森林变量重要性", "数据拟合"])

        if analysis_option == "变量聚类":
            features = st.multiselect("选择用于聚类的变量", df.columns)
            cluster_method = st.selectbox("选择聚类方法", ["K-Means", "层次聚类"])
            n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=10, value=3)

            if features and st.button("开始聚类"):
                X = df[features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                if cluster_method == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                else:
                    model = AgglomerativeClustering(n_clusters=n_clusters)

                labels = model.fit_predict(X_scaled)
                df["Cluster"] = labels
                st.write("聚类结果：", df)

                plt.figure(figsize=(8, 6))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50)
                plt.title("Cluster Distribution (PCA)")
                plt.xlabel("PCA Component 1")
                plt.ylabel("PCA Component 2")
                st.pyplot(plt)

        elif analysis_option == "相关性分析":
            selected_cols = st.multiselect("选择用于相关性分析的变量", df.columns)
            correlation_type = st.radio("选择相关性分析方法", ["Pearson Correlation", "Spearman Correlation"])
            method = "pearson" if correlation_type == "Pearson Correlation" else "spearman"

            if selected_cols:
                correlation_matrix = df[selected_cols].corr(method=method)
                st.write(f"{correlation_type} 矩阵：", correlation_matrix)

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    fmt=".4f",
                    cmap="coolwarm",
                    ax=ax
                )
                ax.set_title(f"{correlation_type} Heatmap")
                st.pyplot(fig)

        elif analysis_option == "灰色关联分析":
            target_col = st.selectbox("选择目标变量", df.columns)
            feature_cols = st.multiselect("选择特征变量", [col for col in df.columns if col != target_col])
            if target_col and feature_cols:
                X = df[feature_cols]
                y = df[target_col]
                normalized_X = (X - X.min()) / (X.max() - X.min())
                normalized_y = (y - y.min()) / (y.max() - y.min())
                grey_relation = normalized_X.apply(lambda x: 1 - abs(x - normalized_y).sum() / len(normalized_y))
                grey_relation_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Grey Relation": grey_relation.values
                }).sort_values(by="Grey Relation", ascending=False)

                st.write("灰色关联分析：", grey_relation_df)

                fig, ax = plt.subplots()
                sns.barplot(x="Grey Relation", y="Feature", data=grey_relation_df, ax=ax)
                ax.set_title("Grey Relation Analysis Results")
                ax.set_xlabel("Grey Relation Coefficient")
                ax.set_ylabel("Features")
                st.pyplot(fig)

        elif analysis_option == "随机森林变量重要性":
            target_col = st.selectbox("选择目标变量列", df.columns)
            feature_cols = st.multiselect("选择特征变量列", [col for col in df.columns if col != target_col])

            if target_col and feature_cols:
                X = df[feature_cols]
                y = df[target_col]
                model = RandomForestRegressor(random_state=42)
                model.fit(X, y)
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False)

                st.write("特征重要性：", importance_df)

                fig, ax = plt.subplots()
                sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
                ax.set_title("Feature Importance (Random Forest)")
                ax.set_xlabel("Importance Score")
                ax.set_ylabel("Features")
                st.pyplot(fig)

        elif analysis_option == "数据拟合":
            x_col = st.selectbox("选择自变量列", df.columns)
            y_col = st.selectbox("选择因变量列", [col for col in df.columns if col != x_col])
            degree = st.slider("选择拟合的多项式阶数", min_value=1, max_value=10, value=1)

            if x_col and y_col:
                X = df[x_col].values
                y = df[y_col].values

                # 删除缺失值
                valid_indices = ~np.isnan(X) & ~np.isnan(y)
                X = X[valid_indices]
                y = y[valid_indices]

                # 多项式拟合
                coefficients = np.polyfit(X, y, degree)
                polynomial = np.poly1d(coefficients)

                # 计算拟合值
                y_fit = polynomial(X)
                r2 = r2_score(y, y_fit)

                # 显示拟合多项式
                st.write(f"拟合的多项式：{polynomial}")
                st.write(f"R²值: {r2:.4f}")

                # 绘图
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X, y, label="Actual Values", alpha=0.7)
                ax.plot(np.sort(X), polynomial(np.sort(X)), color="red", label="Fitted Curve")
                ax.set_title(f"{degree}-Degree Polynomial Fit")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.legend()
                st.pyplot(fig)
