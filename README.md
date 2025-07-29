# Introduction - The Dataset

The **FloodNet** dataset is unique in its genre. It was collected in August 2017, just a few days after Hurricane Harvey hit Louisiana and Texas. The data was gathered using a small UAV platform during the emergency response phase for rescue missions — a critical period where understanding the extent and location of flooding is crucial.

The dataset contains **2,343 images**, divided into:
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

### Semantic Segmentation Labels
The dataset includes the following labels:
- 0) Background  
- 1) Building Flooded  
- 2) Building Non-Flooded  
- 3) Road Flooded  
- 4) Road Non-Flooded  
- 5) Water  
- 6) Tree  
- 7) Vehicle  
- 8) Pool  
- 9) Grass  

Additionally, the dataset contains approximately **4,500 manually created image-question pairs**, averaging **3.5 questions per image**.

For further details, the original paper will be linked at the end of this article.

---

# The Tasks

The FloodNet challenge involves three distinct tasks:

1. **Image Classification** – Determine whether the given image represents a flooded or non-flooded area.  
2. **Image Segmentation** – Assign each pixel of the image to its corresponding class label.  
3. **Visual Question Answering (VQA)** – Answer a set of questions related to the input image.

The results for these tasks can be explored through the accompanying **Streamlit dashboard**.

# Model Testing

To evaluate the trained models, we utilize the **Streamlit dashboard**.  
Follow these steps to test any model:

1. **Launch the Streamlit application** on your local machine:
   ```bash
   streamlit run streamlit_dashboard.py
   ```

2. **Upload the chosen model** (e.g., the trained `.pth` or `.pt` file) using the dashboard’s upload interface.

3. **Select the task** (Image Classification, Segmentation, or VQA).

4. **Upload the input images** for testing or use the provided sample images.

5. The dashboard will display:
   - Predictions (class labels or segmentation masks)
   - Quantitative metrics (e.g., accuracy, mIoU)
   - Visualizations of the model outputs

This interface enables an interactive evaluation of the **Custom CNN** and **ResNet-based models**, making it easier to benchmark their performance across tasks.
