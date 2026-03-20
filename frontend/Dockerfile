# Frontend Dockerfile - Nginx web server
FROM nginx:latest

# Copy HTML file into nginx
COPY index.html /usr/share/nginx/html/index.html

# Copy static assets (CSS, JS)
COPY assets /usr/share/nginx/html/assets

# Copy custom nginx config để cho phép React routing và CORS
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
