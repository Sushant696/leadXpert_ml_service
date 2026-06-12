/**
 * PM2 Ecosystem Config
 * =====================
 * Manages both the Node.js backend AND Python Flask ML service under PM2.
 *
 * Usage:
 *   pm2 start ecosystem.config.js        # start all
 *   pm2 restart ml-service               # restart Flask only
 *   pm2 logs ml-service                  # Flask logs
 *   pm2 save && pm2 startup              # persist across reboots
 */

module.exports = {
  apps: [
    // ── 1. Node.js Backend API
    {
      name:        "leadxpert-api",
      script:      "dist/server.js",          // adjust to your build output
      cwd:         "/var/www/leadxpert/backend",
      instances:   1,
      autorestart: true,
      watch:       false,
      env: {
        NODE_ENV:       "production",
        PORT:           "5000",
        ML_SERVICE_URL: "http://localhost:5001",
        // DB_URI, JWT_SECRET etc. from .env
      },
    },

    // ── 2. Python Flask ML Microservice
    {
      name:        "ml-service",
      script:      "gunicorn",
      args:        "-w 2 -b 0.0.0.0:5001 --timeout 30 app:app",
      cwd:         "/var/www/leadxpert/ml-service",
      interpreter: "/var/www/leadxpert/ml-service/venv/bin/python",
      autorestart: true,
      watch:       false,
      max_memory_restart: "512M",
      env: {
        FLASK_ENV: "production",
        PORT:      "5001",
      },
    },
  ],
};
