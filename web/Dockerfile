# Vite bakes VITE_* env vars into the static bundle at BUILD time, not at
# container start - unlike the Python api service, which reads its env vars
# at runtime. VITE_API_BASE_URL must therefore be a build ARG, and it must
# be a URL the BROWSER (running on the host, not inside the Compose
# network) can reach - "http://api:8000" (Docker-internal DNS) would be
# wrong here even though it's exactly right for the old Streamlit setup,
# which made its backend calls from inside the container instead.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_API_KEY=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_API_KEY=$VITE_API_KEY
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
