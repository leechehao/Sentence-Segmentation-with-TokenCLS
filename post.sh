# json
curl http://127.0.0.1:9489/invocations -X POST -H "Content-Type: application/json" -d '{
    "dataframe_split": {
        "columns": ["text"],
        "data": ["Findings: 1. A 2 cm mass in the right upper lobe, highly suspicious for primary lung cancer. 2. Scattered ground-glass opacities in both lungs, possible early sign of interstitial lung disease. 3. No significant mediastinal lymph node enlargement. 4. Mild pleural effusion on the left side. 5. No evidence of bone metastasis in the visualized portions of the thorax. Conclusion: A. Right upper lobe mass suggestive of lung cancer; biopsy recommended. B. Ground-glass opacities; suggest follow-up CT in 3 months. C. Mild pleural effusion; may require thoracentesis if symptomatic."]
    }
}'

# csv
curl http://127.0.0.1:9489/invocations -X POST -H "Content-Type: text/csv" --data-binary "@post_file.csv"
