# ValueLoop

**ValueLoop** is a Shopify-style analytics tool that helps a store understand and improve **customer lifetime value (LTV)** by turning raw order data into **retention insights** and **clear actions**.

This repo is a **Kaggle dataset demo** (no Shopify API integration yet). It uses the included Olist e-commerce dataset to generate:

- Cohort retention curves (30/60/90-day retention + retention-by-cohort)
- LTV by acquisition month and by segment (category/state)
- A simple churn-risk score and a “next best actions” list (winback, second purchase nudge, VIP loyalty)

## Quickstart

```bash
# Use existing venv/ if present, otherwise create one
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in your terminal.

## Data

By default the dashboard loads the included dataset in `archive/`:

- `archive/olist_orders_dataset.csv`
- `archive/olist_order_items_dataset.csv`
- `archive/olist_products_dataset.csv`
- `archive/olist_customers_dataset.csv`
- `archive/product_category_name_translation.csv`

### Bring your own data (optional)

You can upload a prebuilt `order_fact` CSV in the app. Expected columns:

- `customer_unique_id`
- `order_id`
- `order_purchase_ts` (parseable datetime)
- `order_revenue` (numeric)
- Optional: `primary_category`, `customer_state`

## Project structure

- `app.py`: Streamlit dashboard
- `metrics.py`: data prep + retention/LTV/churn scoring logic
- `archive/`: included Kaggle dataset (Olist)


