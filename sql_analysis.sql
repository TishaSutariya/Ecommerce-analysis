SELECT COUNT(DISTINCT user_id) AS total_users
FROM funnel_data;

SELECT COUNT(DISTINCT user_id) AS purchase_users
FROM funnel_data
WHERE funnel_stage = 'Purchase';

SELECT device,
       COUNT(DISTINCT user_id) AS users
FROM funnel_data
GROUP BY device;

SELECT channel,
       SUM(revenue) AS total_revenue
FROM funnel_data
GROUP BY channel;

SELECT product_category,
       SUM(revenue) AS total_revenue
FROM funnel_data
GROUP BY product_category
ORDER BY total_revenue DESC;