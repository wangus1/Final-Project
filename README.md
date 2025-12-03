
# Predicting Manufacturing Defects (High Defects vs Low Defects)

## 1. Giới thiệu dự án

Dự án này tập trung vào việc **dự đoán khả năng phát sinh lỗi (High Defects)** trong quy trình sản xuất thông qua các chỉ số vận hành như sản lượng, chất lượng nhà cung cấp, bảo trì, tồn kho, năng suất lao động, an toàn, tiêu thụ năng lượng và quy trình Additive Manufacturing.

Mục tiêu cuối cùng là xây dựng một **mô hình Machine Learning giúp phân loại các lô sản xuất thành Low Defects hoặc High Defects**, từ đó hỗ trợ doanh nghiệp:

-   Chủ động kiểm soát chất lượng
    
-   Giảm tỷ lệ lỗi
    
-   Tối ưu chi phí và hiệu suất vận hành
    

----------

## 2. Mục tiêu & Kết quả kỳ vọng

### Mục tiêu

-   Xác định các **yếu tố quan trọng ảnh hưởng đến DefectStatus**, bao gồm:
    
    -   Sản lượng & chi phí sản xuất
        
    -   Chất lượng nhà cung cấp & giao hàng
        
    -   Bảo trì & downtime
        
    -   Quản lý tồn kho
        
    -   Năng suất & an toàn lao động
        
    -   Năng lượng
        
    -   Additive Manufacturing
        
-   Xây dựng mô hình **phân loại DefectStatus (0/1)** nhằm:
    
    -   Phát hiện sớm nguy cơ **High Defects**
        
    -   Hỗ trợ ra quyết định trong sản xuất
        

### Kết quả kỳ vọng

-   Mô hình dự đoán **DefectStatus với độ chính xác cao**
    
-   Bộ **insight hành động (actionable insights)** giúp:
    
    -   Cải thiện chất lượng nhà cung cấp
        
    -   Tối ưu bảo trì
        
    -   Giảm downtime và stockout
        
    -   Nâng cao năng suất lao động
        
-   Xác định **nguyên nhân gốc rễ (root causes)** dẫn đến High Defects
    

----------

## 3. Dataset

### Nguồn dữ liệu

**🏭 Predicting Manufacturing Defects Dataset**

### Mô tả chung

Dataset mô phỏng dữ liệu vận hành trong môi trường sản xuất, phục vụ cho bài toán **dự đoán DefectStatus**. Dữ liệu bao gồm các nhóm chỉ số về:

-   Sản xuất
    
-   Chuỗi cung ứng
    
-   Kiểm soát chất lượng
    
-   Bảo trì
    
-   Tồn kho
    
-   Nhân sự & An toàn
    
-   Năng lượng
    
-   Additive Manufacturing
    

### Cấu trúc dữ liệu

#### Chỉ số sản xuất

-   `ProductionVolume`: Sản lượng mỗi ngày (100 – 1000)
    
-   `ProductionCost`: Chi phí sản xuất mỗi ngày ($5,000 – $20,000)
    

#### Chuỗi cung ứng & Logistics

-   `SupplierQuality`: Điểm chất lượng nhà cung cấp (80% – 100%)
    
-   `DeliveryDelay`: Thời gian giao trễ (0 – 5 ngày)
    

#### Kiểm soát chất lượng

-   `DefectRate`: Lỗi trên 1000 sản phẩm (0.5 – 5.0)
    
-   `QualityScore`: Điểm chất lượng tổng thể (60% – 100%)
    

#### Bảo trì & Downtime

-   `MaintenanceHours`: Giờ bảo trì / tuần (0 – 24)
    
-   `DowntimePercentage`: Tỷ lệ downtime (0% – 5%)
    

#### Quản lý tồn kho

-   `InventoryTurnover`: Vòng quay tồn kho (2 – 10)
    
-   `StockoutRate`: Tỷ lệ thiếu hàng (0% – 10%)
    

#### Năng suất & An toàn

-   `WorkerProductivity`: Mức năng suất lao động (80% – 100%)
    
-   `SafetyIncidents`: Số sự cố an toàn / tháng (0 – 10)
    

#### Năng lượng

-   `EnergyConsumption`: Lượng tiêu thụ điện (1000 – 5000 kWh)
    
-   `EnergyEfficiency`: Hệ số hiệu quả năng lượng (0.1 – 0.5)
    

#### Additive Manufacturing

-   `AdditiveProcessTime`: Thời gian xử lý additive (1 – 10 giờ)
    
-   `AdditiveMaterialCost`: Chi phí vật liệu additive ($100 – $500)
    

#### Biến mục tiêu (Target)

-   `DefectStatus`:
    
    -   `0` = Low Defects
        
    -   `1` = High Defects
        

----------

## 4.  KẾ HOẠCH PHÂN TÍCH CHI TIẾT



### 4.1. Data Cleaning & Data Validation

Mục tiêu: Đảm bảo dữ liệu **sạch – đúng – sẵn sàng cho phân tích và mô hình hóa**.

Các bước thực hiện:

-   Kiểm tra **missing values** trên toàn bộ các biến:
    
    -   Nếu có: xử lý bằng phương pháp phù hợp (mean/median hoặc loại bỏ).
        
-   Chuẩn hóa **kiểu dữ liệu (data types)**:
    
    -   Chuyển các cột số về đúng định dạng `int` hoặc `float`.
        
-   Kiểm tra **dòng trùng lặp (duplicated rows)**:
    
    -   Loại bỏ để tránh làm sai lệch mô hình.
        
-   **Ràng buộc logic dữ liệu (data validation)**:
    
    -   Đảm bảo các biến nằm trong phạm vi hợp lý đã mô tả ở phần Dataset:
        
        -   `SupplierQuality` ∈ [80, 100]
            
        -   `DowntimePercentage` ∈ [0, 5]
            
        -   `DefectRate` ∈ [0.5, 5.0], …
            
-   Phát hiện **outliers** bằng:
    
    -   IQR
        
    -   Boxplot
        
-   Đánh giá ảnh hưởng của outliers:
    
    -   Giữ lại nếu mang ý nghĩa thực tế vận hành.
        

**Kết quả mong muốn:** Một bộ dữ liệu sạch, nhất quán, không nhiễu logic để đưa vào EDA và Modeling.

----------

### 4.2. Exploratory Data Analysis (EDA)

Mục tiêu: **Hiểu rõ đặc điểm dữ liệu và hành vi của nhóm High Defects vs Low Defects.**

**Thống kê mô tả (Descriptive Statistics):**

-   Mean, median, std cho từng biến.
    
-   So sánh thống kê giữa:
    
    -   `DefectStatus = 0`
        
    -   `DefectStatus = 1`
        

**Phân tích phân phối (Distribution Analysis):**

-   Histogram cho:
    
    -   ProductionVolume
        
    -   DefectRate
        
    -   MaintenanceHours
        
    -   StockoutRate
        
    -   EnergyConsumption
        
-   Kiểm tra dữ liệu lệch phải / lệch trái.
    

**So sánh theo DefectStatus (Group Comparison):**

-   Boxplot:
    
    -   MaintenanceHours vs DefectStatus
        
    -   SupplierQuality vs DefectStatus
        
    -   StockoutRate vs DefectStatus
        
    -   WorkerProductivity vs DefectStatus
        
-   Mục tiêu: Tìm **biến có sự khác biệt rõ rệt giữa hai nhóm lỗi**.
    

📌 **Output EDA mong muốn:**

-   Xác định nhóm biến:
    
    -   Liên quan mạnh đến High Defects
        
    -   Hầu như không ảnh hưởng
        

----------

### 4.3. Phân tích tương quan & chọn feature

-   Tính **Correlation Matrix** cho toàn bộ biến số.
    
-   Vẽ **Heatmap tương quan**:
    
    -   Phát hiện các cặp biến:
        
        -   Có tương quan cao với `DefectStatus`
            
        -   Có nguy cơ **đa cộng tuyến (multicollinearity)**.
            
-   Lọc ra các biến có |correlation| cao với target, ví dụ:
    
    -   `DefectRate`
        
    -   `MaintenanceHours`
        
    -   `SupplierQuality`
        
    -   `QualityScore`
        
    -   `StockoutRate`
        
    -   `WorkerProductivity`
        

**Mục tiêu:** Rút gọn bộ biến đầu vào giúp mô hình:

-   Ổn định hơn
    
-   Tránh overfitting
    
-   Dễ diễn giải hơn cho doanh nghiệp
    

----------

### 4.4. Modeling – Xây dựng mô hình Logistic Regression

**Lý do chọn Logistic Regression:**

-   Phù hợp bài toán **phân loại nhị phân**
    
-   Dễ **giải thích hệ số (interpretability)**
    
-   Phù hợp với dữ liệu vận hành doanh nghiệp
    

**Quy trình:**

1.  Tách dữ liệu:
    
    -   Train set
        
    -   Test set
        
2.  Chuẩn hóa dữ liệu nếu cần (StandardScaler)
    
3.  Huấn luyện mô hình Logistic Regression
    
4.  Đánh giá mô hình bằng:
    
    -   Accuracy
        
    -   Precision
        
    -   Recall
        
    -   F1-score
        
    -   Confusion Matrix
        
5.  Phân tích **hệ số hồi quy (coefficients)** để hiểu mức độ ảnh hưởng của từng biến.
    

**Kết quả mong muốn:**

-   Mô hình đủ tốt để:
    
    -   Phát hiện sớm High Defects
        
    -   Hỗ trợ ra quyết định trong sản xuất
