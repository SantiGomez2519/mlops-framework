<template>
  <div class="container">
    <header>
      <h1>House Price Predictor</h1>
      <p class="subtitle">Enter house details to get a price estimate</p>
    </header>

    <form @submit.prevent="predict" class="form">
      <div class="form-grid">
        <div class="form-group">
          <label for="sqft">Square Footage</label>
          <input id="sqft" v-model.number="form.sqft" type="number" min="1" required placeholder="e.g. 2000" />
        </div>

        <div class="form-group">
          <label for="bedrooms">Bedrooms</label>
          <input id="bedrooms" v-model.number="form.bedrooms" type="number" min="1" required placeholder="e.g. 3" />
        </div>

        <div class="form-group">
          <label for="bathrooms">Bathrooms</label>
          <input id="bathrooms" v-model.number="form.bathrooms" type="number" min="0.5" step="0.5" required placeholder="e.g. 2.5" />
        </div>

        <div class="form-group">
          <label for="year_built">Year Built</label>
          <input id="year_built" v-model.number="form.year_built" type="number" min="1900" :max="currentYear" required placeholder="e.g. 1995" />
        </div>

        <div class="form-group">
          <label for="location">Location</label>
          <select id="location" v-model="form.location" required>
            <option value="" disabled>Select location</option>
            <option v-for="loc in locations" :key="loc" :value="loc">{{ loc }}</option>
          </select>
        </div>

        <div class="form-group">
          <label for="condition">Condition</label>
          <select id="condition" v-model="form.condition" required>
            <option value="" disabled>Select condition</option>
            <option value="Poor">Poor</option>
            <option value="Fair">Fair</option>
            <option value="Good">Good</option>
            <option value="Excellent">Excellent</option>
          </select>
        </div>
      </div>

      <button type="submit" :disabled="loading" class="btn-predict">
        <span v-if="loading">Predicting...</span>
        <span v-else>Predict Price</span>
      </button>
    </form>

    <div v-if="error" class="error">{{ error }}</div>

    <div v-if="result" class="result">
      <h2>Predicted Price</h2>
      <p class="price">{{ formatPrice(result.predicted_price) }}</p>
      <p class="meta">Model: {{ result.model_name }} | R² Score: {{ (result.r2_score * 100).toFixed(1) }}%</p>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      form: {
        sqft: null,
        bedrooms: null,
        bathrooms: null,
        year_built: null,
        location: '',
        condition: '',
      },
      locations: [],
      result: null,
      error: null,
      loading: false,
      currentYear: new Date().getFullYear(),
    }
  },
  mounted() {
    fetch('/model/locations')
      .then(res => res.json())
      .then(data => { this.locations = data.locations })
      .catch(() => {
        this.locations = ['Downtown', 'Mountain', 'Rural', 'Suburb', 'Urban', 'Waterfront']
      })
  },
  methods: {
    async predict() {
      this.loading = true
      this.error = null
      this.result = null

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.form),
        })

        if (!res.ok) {
          const err = await res.json()
          throw new Error(err.detail || 'Prediction failed')
        }

        this.result = await res.json()
      } catch (e) {
        this.error = e.message
      } finally {
        this.loading = false
      }
    },
    formatPrice(value) {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0,
      }).format(value)
    },
  },
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: #f0f2f5;
  color: #1a1a2e;
  min-height: 100vh;
}

.container {
  max-width: 640px;
  margin: 0 auto;
  padding: 40px 20px;
}

header {
  text-align: center;
  margin-bottom: 32px;
}

header h1 {
  font-size: 28px;
  font-weight: 700;
  color: #1a1a2e;
}

.subtitle {
  color: #6b7280;
  margin-top: 4px;
  font-size: 14px;
}

.form {
  background: #fff;
  border-radius: 12px;
  padding: 28px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.form-group {
  display: flex;
  flex-direction: column;
}

.form-group label {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 6px;
  color: #374151;
}

.form-group input,
.form-group select {
  padding: 10px 12px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
  color: #1a1a2e;
  background: #f9fafb;
  transition: border-color 0.2s;
}

.form-group input:focus,
.form-group select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.btn-predict {
  width: 100%;
  margin-top: 20px;
  padding: 12px;
  background: #3b82f6;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 15px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-predict:hover:not(:disabled) {
  background: #2563eb;
}

.btn-predict:disabled {
  background: #93c5fd;
  cursor: not-allowed;
}

.error {
  margin-top: 16px;
  padding: 12px 16px;
  background: #fef2f2;
  color: #dc2626;
  border-radius: 8px;
  font-size: 14px;
}

.result {
  margin-top: 24px;
  background: #fff;
  border-radius: 12px;
  padding: 28px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  text-align: center;
}

.result h2 {
  font-size: 14px;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.price {
  font-size: 40px;
  font-weight: 700;
  color: #059669;
  margin: 8px 0;
}

.meta {
  font-size: 13px;
  color: #9ca3af;
}
</style>
