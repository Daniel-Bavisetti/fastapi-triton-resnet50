import http from "k6/http";
import { check } from "k6";
import { Trend } from "k6/metrics";

const predictLatency = new Trend("predict_latency_ms");

export const options = {
  vus: 10,
  duration: "30s",
  thresholds: {
    http_req_duration: ["p(95)<2000"],
    http_req_failed: ["rate<0.05"],
  },
};

const imagePath = __ENV.TEST_IMAGE || "./sample.jpg";
const imageData = open(imagePath, "b");

export default function () {
  const payload = {
    image: http.file(imageData, "sample.jpg", "image/jpeg"),
  };

  const res = http.post("http://localhost:8080/predict", payload);
  predictLatency.add(res.timings.duration);

  check(res, {
    "status is 200": (r) => r.status === 200,
    "response has success": (r) => {
      try {
        return JSON.parse(r.body).status === "success";
      } catch (_) {
        return false;
      }
    },
  });
}

export function handleSummary(data) {
  const rps = data.metrics.http_reqs.rate;
  const p95 = data.metrics.http_req_duration["p(95)"];
  const failRate = data.metrics.http_req_failed.rate;

  const summary = [
    "\n=== k6 Summary ===",
    `RPS: ${rps.toFixed(2)}`,
    `Latency p95 (ms): ${p95.toFixed(2)}`,
    `Failure rate: ${(failRate * 100).toFixed(2)}%`,
    "==================\n",
  ].join("\n");

  return {
    stdout: summary,
  };
}
