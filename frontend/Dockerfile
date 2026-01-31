# MediMind BTIS - Frontend Dockerfile
# Multi-stage build for React/Vite application

# Stage 1: Build stage
FROM node:20-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --silent

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Stage 2: Production stage with Nginx
FROM nginx:alpine as production

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Add non-root user
RUN addgroup -g 1001 -S medimind && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nologin -G medimind medimind && \
    chown -R medimind:medimind /var/cache/nginx /var/run /var/log/nginx /usr/share/nginx/html

# Create nginx pid directory
RUN mkdir -p /var/run/nginx && chown medimind:medimind /var/run/nginx

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# Expose port
EXPOSE 80

# Switch to non-root user
USER medimind

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
