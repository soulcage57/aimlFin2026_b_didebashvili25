# DDoS ATTACK DETECTION
## Web Server Log Analysis: DDoS Attack Detection1
### Introduction
This report documents the detection and characterization of a Distributed Denial of Service (DDoS) attack using polynomial regression analysis applied to web server access logs. The analysis covers a one-hour observation window and identifies the precise time interval during which the attack occurred.
### Dataset Description
The web server log file follows the Apache/Nginx Combined Log Format and contains 53,235 individual HTTP request records spanning exactly one hour. Each log line contains the source IP address, timestamp with timezone, HTTP method and endpoint, response status code, response size in bytes, referer, and user-agent string. For this analysis, the timestamp and source IP were the primary fields of interest.
 ### Methodology
 The analysis pipeline consists of four sequential stages: log parsing, baseline modeling via polynomial regression, anomaly detection through residual analysis, and visualization.
 
**Log Parsing and Aggregation** \
A regular expression extracts the source IP address and full timestamp from each log line. Requests are then aggregated into one-minute buckets by truncating the timestamp to the hour:minute level. This produces a time series of 61 data points representing request counts per minute.
**Polynomial Regression Baseline** \
A polynomial regression model of degree 15 is fitted to the per-minute request count time series. The minute index (0–60) serves as the independent variable X; the request count is the dependent variable Y. The fitted curve represents the expected 'normal' traffic pattern across the hour — capturing gradual trends such as traffic ramp-up, sustained load, and gradual decrease.

The polynomial degree of 15 was selected to provide sufficient flexibility to model hour-scale traffic variations without overfitting to individual minute-level noise. The resulting fit achieved R² = 0.557, indicating that the model explains approximately 56% of the total variance — adequate for a baseline that will be used only to identify gross deviations.

**Residual Analysis and Anomaly Detection** \
The residual for each minute is computed as the difference between the observed count and the fitted baseline value. Under normal conditions, residuals should be approximately normally distributed around zero. A DDoS attack produces a massive positive spike that exceeds the residuals of normal minutes by many standard deviations.

An anomaly threshold is set at the mean residual plus 2.5 standard deviations (μ + 2.5σ). Any minute whose residual exceeds this threshold is flagged as anomalous. Consecutive (or near-consecutive, gap ≤ 2 minutes) anomalous minutes are grouped into a single attack interval.
## Results
**DDoS Attack Interval: 18:07 – 18:08 (UTC+4) 9,482 requests over 2 minutes — 6.4× above normal baseline**

<img width="823" height="370" alt="image" src="https://github.com/user-attachments/assets/71360e03-bec6-4c0c-9de5-5fedb73442a8" />


### Time Series Summary
<img width="622" height="274" alt="image" src="https://github.com/user-attachments/assets/0b8f482f-41b6-4584-9183-4a5c6847ed6f" />

### Source IP Analysis
During the attack window, the server received requests from 124 unique IP addresses at 18:07 and 92 at 18:08. While this is somewhat elevated compared to a typical minute (average ~68 unique IPs), the attack is primarily characterized by a dramatically higher request rate per IP — indicative of a volumetric flood from a botnet or amplification attack, rather than a pure distributed low-rate attack.
### Visualizations
<img width="938" height="834" alt="image" src="https://github.com/user-attachments/assets/9ce322fc-f95d-418e-a714-90f60de5d401" />

## main fragments of the source code
 **Log Parsing** 
 ```
pattern = r'(\d+\.\d+\.\d+\.\d+).*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
per_minute = defaultdict(int)

with open('log.txt', 'r') as f:
    for line in f:
        m = re.search(pattern, line)
        if m:
            ts = datetime.strptime(m.group(2), '%Y-%m-%d %H:%M:%S')
            per_minute[ts.strftime('%H:%M')] += 1
```
**Polynomial Regression Baseline**
```
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = np.arange(len(counts)).reshape(-1, 1)
model = make_pipeline(PolynomialFeatures(15), LinearRegression())
model.fit(x, counts)
baseline  = model.predict(x)
residuals = counts - baseline   # observed − fitted
r2        = r2_score(counts, baseline)  # R² = 0.557
```
**Anomaly Detection**
```
mu, sigma    = np.mean(residuals), np.std(residuals)
threshold    = mu + 2.5 * sigma       # detection boundary
anomaly_mask = residuals > threshold  # True for attack minutes

# Group consecutive anomalous minutes into attack intervals
groups, start, prev = [], anomaly_idx[0], anomaly_idx[0]
for idx in anomaly_idx[1:]:
    if idx - prev <= 2:
        prev = idx
    else:
        groups.append((start, prev))
        start = prev = idx
groups.append((start, prev))
```
