/* ScoutIQ — dashboard.js
   Shared dashboard utilities and interactive enhancements */

'use strict';

// ── Animated counter ──────────────────────────────────────────────────────────
function animateCounter(el, target, duration = 1200, isFloat = false) {
  const start = performance.now();
  const startVal = 0;

  function update(timestamp) {
    const elapsed = timestamp - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    const current = startVal + (target - startVal) * eased;

    el.textContent = isFloat
      ? current.toFixed(2)
      : Math.floor(current).toLocaleString();

    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// Run counter animation on all kpi-value elements that contain numbers
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.kpi-value').forEach(el => {
    const raw = parseFloat(el.textContent.replace(/[^0-9.]/g, ''));
    if (!isNaN(raw) && raw > 0) {
      const isFloat = el.textContent.includes('.');
      el.dataset.target = raw;
      animateCounter(el, raw, 1400, isFloat);
    }
  });
});

// ── Scroll-triggered fade-in ──────────────────────────────────────────────────
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.card, .chart-container').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(16px)';
  el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
  observer.observe(el);
});

// ── Tooltip on data table rows ────────────────────────────────────────────────
document.querySelectorAll('.data-table tbody tr').forEach(row => {
  row.style.cursor = 'pointer';
});

// ── Chart.js global defaults ──────────────────────────────────────────────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(13,20,33,0.95)';
  Chart.defaults.plugins.tooltip.borderColor = 'rgba(255,255,255,0.1)';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 12;
  Chart.defaults.plugins.tooltip.titleColor = '#f1f5f9';
  Chart.defaults.plugins.tooltip.bodyColor = '#94a3b8';
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
}
