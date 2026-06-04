-- USER FUNNEL ANALYSIS - SQLite-adapted (simplified)

-- 1. FUNNEL STAGE BREAKDOWN (simplified; no ROW_NUMBER)
CREATE TEMP VIEW funnel_stages AS
SELECT
    CASE
        WHEN event_type = 'App Open' THEN 1
        WHEN event_type = 'Browse' THEN 2
        WHEN event_type = 'Product View' THEN 3
        WHEN event_type = 'Add to Cart' THEN 4
        WHEN event_type = 'Checkout' THEN 5
        WHEN event_type = 'Payment' THEN 6
        WHEN event_type = 'Purchase' THEN 7
        ELSE NULL
    END AS funnel_stage,
    event_type AS stage_name,
    user_id,
    session_id
FROM user_events
WHERE event_type IN ('App Open','Browse','Product View','Add to Cart','Checkout','Payment','Purchase');

SELECT
    funnel_stage,
    stage_name,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(DISTINCT session_id) AS total_sessions
FROM funnel_stages
GROUP BY funnel_stage, stage_name
ORDER BY funnel_stage;

-- 2. DROP-OFF ANALYSIS BY STAGE (same logic)
CREATE TEMP VIEW stage_funnel AS
SELECT 
    session_id,
    MAX(CASE WHEN event_type = 'App Open' THEN 1 ELSE 0 END) AS saw_app_open,
    MAX(CASE WHEN event_type = 'Browse' THEN 1 ELSE 0 END) AS saw_browse,
    MAX(CASE WHEN event_type = 'Product View' THEN 1 ELSE 0 END) AS saw_product,
    MAX(CASE WHEN event_type = 'Add to Cart' THEN 1 ELSE 0 END) AS saw_cart,
    MAX(CASE WHEN event_type = 'Checkout' THEN 1 ELSE 0 END) AS saw_checkout,
    MAX(CASE WHEN event_type = 'Payment' THEN 1 ELSE 0 END) AS saw_payment,
    MAX(CASE WHEN event_type = 'Purchase' THEN 1 ELSE 0 END) AS saw_purchase
FROM user_events
GROUP BY session_id;

SELECT
    'App Open → Browse' AS transition,
    SUM(saw_app_open) AS users_at_start,
    SUM(saw_browse) AS users_at_end,
    SUM(CASE WHEN saw_app_open = 1 AND saw_browse = 0 THEN 1 ELSE 0 END) AS dropped_off,
    ROUND(100.0 * SUM(CASE WHEN saw_app_open = 1 AND saw_browse = 0 THEN 1 ELSE 0 END) / NULLIF(SUM(saw_app_open),0),2) AS drop_off_percentage
FROM stage_funnel
UNION ALL
SELECT 'Browse → Product View',
    SUM(saw_browse),
    SUM(saw_product),
    SUM(CASE WHEN saw_browse = 1 AND saw_product = 0 THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(CASE WHEN saw_browse = 1 AND saw_product = 0 THEN 1 ELSE 0 END) / NULLIF(SUM(saw_browse),0),2)
FROM stage_funnel
UNION ALL
SELECT 'Product View → Add to Cart',
    SUM(saw_product),
    SUM(saw_cart),
    SUM(CASE WHEN saw_product = 1 AND saw_cart = 0 THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(CASE WHEN saw_product = 1 AND saw_cart = 0 THEN 1 ELSE 0 END) / NULLIF(SUM(saw_product),0),2)
FROM stage_funnel
UNION ALL
SELECT 'Add to Cart → Checkout',
    SUM(saw_cart),
    SUM(saw_checkout),
    SUM(CASE WHEN saw_cart = 1 AND saw_checkout = 0 THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(CASE WHEN saw_cart = 1 AND saw_checkout = 0 THEN 1 ELSE 0 END) / NULLIF(SUM(saw_cart),0),2)
FROM stage_funnel
UNION ALL
SELECT 'Checkout → Payment',
    SUM(saw_checkout),
    SUM(saw_payment),
    SUM(CASE WHEN saw_checkout = 1 AND saw_payment = 0 THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(CASE WHEN saw_checkout = 1 AND saw_payment = 0 THEN 1 ELSE 0 END) / NULLIF(SUM(saw_checkout),0),2)
FROM stage_funnel
UNION ALL
SELECT 'Payment → Purchase',
    SUM(saw_payment),
    SUM(saw_purchase),
    SUM(CASE WHEN saw_payment = 1 AND saw_purchase = 0 THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(CASE WHEN saw_payment = 1 AND saw_purchase = 0 THEN 1 ELSE 0 END) / NULLIF(SUM(saw_payment),0),2)
FROM stage_funnel;

-- 3. DEVICE PERFORMANCE (works in SQLite)
SELECT 
    device_type,
    COUNT(DISTINCT user_id) AS total_users,
    COUNT(DISTINCT session_id) AS total_sessions,
    COUNT(DISTINCT CASE WHEN conversion_flag = 1 THEN user_id END) AS converters,
    ROUND(100.0 * COUNT(DISTINCT CASE WHEN conversion_flag = 1 THEN user_id END) / NULLIF(COUNT(DISTINCT user_id),0), 2) AS conversion_rate,
    ROUND(AVG(session_duration_seconds), 0) AS avg_session_duration,
    ROUND(COUNT(DISTINCT session_id) * 1.0 / NULLIF(COUNT(DISTINCT user_id),0), 2) AS sessions_per_user,
    ROUND(SUM(CASE WHEN conversion_flag = 1 THEN product_price ELSE 0 END) * 1.0 / NULLIF(COUNT(DISTINCT CASE WHEN conversion_flag = 1 THEN user_id END),0), 2) AS avg_order_value
FROM user_events
GROUP BY device_type
ORDER BY conversion_rate DESC;