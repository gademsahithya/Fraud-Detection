/* ------------------------------------------------------------------
   Fraud screening UI  -  front-end logic
   Sends the form to  POST /predict  and draws the result.
------------------------------------------------------------------- */
const form       = document.getElementById("txForm");
const submitBtn  = document.getElementById("submitBtn");
const errorBox   = document.getElementById("formError");
const card       = document.getElementById("resultCard");
const gaugeFill  = document.getElementById("gaugeFill");
const gaugeTick  = document.getElementById("gaugeTick");
const gaugeValue = document.getElementById("gaugeValue");
const verdict    = document.getElementById("verdict");
const verdictTxt = document.getElementById("verdictText");
const seen       = document.getElementById("seen");
const seenList   = document.getElementById("seenList");

const THRESHOLD = window.FRAUD_THRESHOLD ?? 0.4;

/* ---------- example transactions ---------- */
const SAMPLES = {
  normal: {
    type: "PAYMENT", amount: 1864, step: 12,
    oldbalanceOrg: 21249, newbalanceOrig: 19385,
    oldbalanceDest: 0, newbalanceDest: 0,
    origin_transaction_count: 6, origin_avg_amount: 2210, origin_max_amount: 4900,
    destination_transaction_count: 340, destination_unique_origins: 310,
  },
  fraud: {
    type: "TRANSFER", amount: 181000, step: 1,
    oldbalanceOrg: 181000, newbalanceOrig: 0,
    oldbalanceDest: 0, newbalanceDest: 0,
    origin_transaction_count: 0, origin_avg_amount: 0, origin_max_amount: 0,
    destination_transaction_count: 0, destination_unique_origins: 0,
  },
};

document.querySelectorAll("[data-sample]").forEach((btn) => {
  btn.addEventListener("click", () => {
    fillForm(SAMPLES[btn.dataset.sample]);
    form.requestSubmit();
  });
});

function fillForm(values) {
  Object.entries(values).forEach(([name, value]) => {
    if (name === "type") {
      form.querySelector(`input[name="type"][value="${value}"]`).checked = true;
    } else {
      form.elements[name].value = value;
    }
  });
  clearErrors();
}

/* ---------- gauge ---------- */
// Point on the semicircle for a value between 0 and 1.
function gaugePoint(f, radius) {
  const angle = Math.PI * f;
  return [110 - radius * Math.cos(angle), 110 - radius * Math.sin(angle)];
}

function placeThresholdTick() {
  const [x1, y1] = gaugePoint(THRESHOLD, 78);
  const [x2, y2] = gaugePoint(THRESHOLD, 102);
  gaugeTick.setAttribute("x1", x1); gaugeTick.setAttribute("y1", y1);
  gaugeTick.setAttribute("x2", x2); gaugeTick.setAttribute("y2", y2);
}
placeThresholdTick();

function setGauge(probability) {
  // pathLength is 100, so the dash length is simply a percentage
  gaugeFill.style.strokeDasharray = `${probability * 100} 100`;
}

function formatPercent(p) {
  const pct = p * 100;
  if (pct > 0 && pct < 0.1) return "<0.1%";
  return (pct >= 10 ? pct.toFixed(0) : pct.toFixed(1)) + "%";
}

/* ---------- validation ---------- */
function clearErrors() {
  errorBox.hidden = true;
  form.querySelectorAll(".invalid").forEach((el) => el.classList.remove("invalid"));
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function validate() {
  clearErrors();
  const bad = [...form.querySelectorAll('input[type="number"]')].filter(
    (el) => el.value === "" || Number(el.value) < 0 || Number.isNaN(Number(el.value))
  );
  bad.forEach((el) => el.classList.add("invalid"));
  if (bad.length) {
    showError("Enter a number of 0 or more in every highlighted field.");
    bad[0].focus();
    return false;
  }
  return true;
}

/* ---------- submit ---------- */
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!validate()) return;

  const payload = Object.fromEntries(new FormData(form).entries());

  submitBtn.disabled = true;
  submitBtn.textContent = "Checking…";
  card.dataset.state = "loading";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The server rejected this request.");
    renderResult(data);
  } catch (err) {
    card.dataset.state = "idle";
    setGauge(0);
    gaugeValue.textContent = "–";
    verdict.textContent = "Could not check this transaction";
    verdictTxt.textContent = err.message.includes("fetch")
      ? "The server is not reachable. Make sure app.py is running."
      : err.message;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Check transaction";
  }
});

form.addEventListener("reset", () => {
  clearErrors();
  card.dataset.state = "idle";
  setGauge(0);
  gaugeValue.textContent = "–";
  verdict.textContent = "No result yet";
  verdictTxt.textContent = "Fill in the form or pick an example, then check the transaction.";
  seen.hidden = true;
});

/* ---------- render ---------- */
const COPY = {
  low: {
    title: "Looks legitimate",
    text: "The model sees little in common with known fraud. No action needed.",
  },
  elevated: {
    title: "Some risk signals",
    text: "Below the flagging threshold, but worth a second look if the account is sensitive.",
  },
  high: {
    title: "Likely fraud",
    text: "This is above the flagging threshold. Hold the transaction and send it for manual review.",
  },
};

const FEATURE_LABELS = {
  step: "Hour of simulation",
  amount: "Amount",
  origin_transaction_count: "Sender past transactions",
  origin_avg_amount: "Sender average amount",
  origin_max_amount: "Sender largest amount",
  amount_vs_origin_avg: "Amount vs sender average",
  destination_unique_origins: "Receiver distinct senders",
  destination_transaction_count: "Receiver past transactions",
  origin_amount_ratio: "Amount vs sender balance",
  origin_remaining_ratio: "Sender balance left",
  destination_amount_ratio: "Amount vs receiver balance",
  destination_balance_ratio: "Receiver balance change",
  type_CASH_IN: "Type: cash in",
  type_CASH_OUT: "Type: cash out",
  type_DEBIT: "Type: debit",
  type_PAYMENT: "Type: payment",
  type_TRANSFER: "Type: transfer",
};

function renderResult(data) {
  const copy = COPY[data.level];
  card.dataset.state = data.level;
  setGauge(data.probability);
  gaugeValue.textContent = formatPercent(data.probability);
  verdict.textContent = copy.title;
  verdictTxt.textContent = copy.text;

  seenList.innerHTML = "";
  Object.entries(data.features).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "pair";
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = FEATURE_LABELS[key] || key;
    dd.textContent = Number(value).toLocaleString();
    row.append(dt, dd);
    seenList.append(row);
  });
  seen.hidden = false;
}
