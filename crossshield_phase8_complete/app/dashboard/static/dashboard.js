const hoursSelect = document.getElementById("hours");
const refreshButton = document.getElementById("refreshButton");
const errorMessage = document.getElementById("errorMessage");

async function fetchJSON(url) {
    const response = await fetch(url);

    if (!response.ok) {
        let message = "Dashboard data is unavailable.";

        try {
            const data = await response.json();

            if (data.detail) {
                message = data.detail;
            }
        } catch (_) {
            // Keep the default message.
        }

        throw new Error(message);
    }

    return response.json();
}


function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}


function hideError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}


function setText(id, value) {
    const element = document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}


function formatRisk(value) {
    if (value === null || value === undefined) {
        return "-";
    }

    return Number(value).toFixed(3);
}


function formatLatency(value) {
    if (value === null || value === undefined) {
        return "-";
    }

    return `${Number(value).toFixed(2)} ms`;
}


function renderDistribution(elementId, distribution) {
    const container = document.getElementById(elementId);

    container.innerHTML = "";

    const entries = Object.entries(distribution || {});

    if (entries.length === 0) {
        container.innerHTML = '<div class="empty">No data available</div>';
        return;
    }

    entries.sort((a, b) => b[1] - a[1]);

    for (const [label, value] of entries) {
        const item = document.createElement("div");
        item.className = "distribution-item";

        const labelElement = document.createElement("span");
        labelElement.className = "distribution-label";
        labelElement.textContent = label;

        const valueElement = document.createElement("span");
        valueElement.className = "distribution-value";
        valueElement.textContent = value;

        item.appendChild(labelElement);
        item.appendChild(valueElement);

        container.appendChild(item);
    }
}


function renderTopCategories(elementId, categories) {
    const container = document.getElementById(elementId);

    container.innerHTML = "";

    if (!categories || categories.length === 0) {
        container.innerHTML = '<div class="empty">No data available</div>';
        return;
    }

    for (const entry of categories) {
        const label = entry[0];
        const value = entry[1];

        const item = document.createElement("div");
        item.className = "distribution-item";

        const labelElement = document.createElement("span");
        labelElement.className = "distribution-label";
        labelElement.textContent = label;

        const valueElement = document.createElement("span");
        valueElement.className = "distribution-value";
        valueElement.textContent = value;

        item.appendChild(labelElement);
        item.appendChild(valueElement);

        container.appendChild(item);
    }
}


function getBadgeClass(value) {
    if (!value) {
        return "badge-neutral";
    }

    const normalized = value.toUpperCase();

    if (
        normalized === "ALLOW" ||
        normalized === "COMPLETED"
    ) {
        return "badge-allow";
    }

    if (
        normalized === "ALLOW_MONITOR" ||
        normalized === "MEDIUM"
    ) {
        return "badge-monitor";
    }

    if (
        normalized === "BLOCK" ||
        normalized.includes("BLOCK")
    ) {
        return "badge-block";
    }

    if (
        normalized === "FLAG_VERIFY" ||
        normalized === "HIGH"
    ) {
        return "badge-flag";
    }

    return "badge-neutral";
}


function createBadge(value) {
    const badge = document.createElement("span");

    badge.className = `badge ${getBadgeClass(value)}`;
    badge.textContent = value || "-";

    return badge;
}


function renderEvents(events) {
    const tableBody = document.getElementById("eventsTable");

    tableBody.innerHTML = "";

    const eventCount = document.getElementById("eventCount");
    eventCount.textContent = `${events.length} events`;

    if (!events || events.length === 0) {
        const row = document.createElement("tr");

        const cell = document.createElement("td");

        cell.colSpan = 8;
        cell.className = "empty";
        cell.textContent = "No security events found.";

        row.appendChild(cell);
        tableBody.appendChild(row);

        return;
    }

    for (const event of events) {
        const row = document.createElement("tr");

        const timeCell = document.createElement("td");
        timeCell.textContent = formatDate(event.occurred_at);

        const requestCell = document.createElement("td");
        requestCell.textContent = shortenId(event.request_id);

        const languageCell = document.createElement("td");
        languageCell.textContent = event.language || "-";

        const riskCell = document.createElement("td");
        riskCell.textContent = formatRisk(event.overall_risk);

        const decisionCell = document.createElement("td");
        decisionCell.appendChild(
            createBadge(event.input_decision)
        );

        const llmCell = document.createElement("td");

        if (event.llm_called) {
            llmCell.textContent =
                event.llm_provider || "Called";
        } else {
            llmCell.textContent = "Not called";
        }

        const outputCell = document.createElement("td");

        if (!event.output_safety_checked) {
            outputCell.appendChild(
                createBadge("NOT_CHECKED")
            );
        } else {
            outputCell.appendChild(
                createBadge(
                    event.output_safety_decision || "-"
                )
            );
        }

        const outcomeCell = document.createElement("td");
        outcomeCell.appendChild(
            createBadge(event.outcome)
        );

        row.appendChild(timeCell);
        row.appendChild(requestCell);
        row.appendChild(languageCell);
        row.appendChild(riskCell);
        row.appendChild(decisionCell);
        row.appendChild(llmCell);
        row.appendChild(outputCell);
        row.appendChild(outcomeCell);

        tableBody.appendChild(row);
    }
}


function renderTrends(trends) {
    const container =
        document.getElementById("trendContainer");

    container.innerHTML = "";

    const status =
        document.getElementById("trendStatus");

    if (!trends || trends.length === 0) {
        status.textContent = "No data";
        container.innerHTML =
            '<div class="empty">No trend data available.</div>';
        return;
    }

    status.textContent =
        `${trends.length} hourly buckets`;

    const maximum = Math.max(
        ...trends.map(item => item.request_count),
        1
    );

    for (const point of trends) {
        const wrapper =
            document.createElement("div");

        wrapper.className =
            "trend-bar-wrapper";

        const bar =
            document.createElement("div");

        bar.className = "trend-bar";

        const height =
            Math.max(
                (point.request_count / maximum) * 160,
                4
            );

        bar.style.height = `${height}px`;

        bar.title =
            `${point.request_count} requests | ` +
            `Risk: ${formatRisk(point.average_overall_risk)}`;

        const label =
            document.createElement("div");

        label.className = "trend-label";

        label.textContent =
            formatHour(point.bucket);

        wrapper.appendChild(bar);
        wrapper.appendChild(label);

        container.appendChild(wrapper);
    }
}


function formatDate(value) {
    if (!value) {
        return "-";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return date.toLocaleString();
}


function formatHour(value) {
    if (!value) {
        return "";
    }

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return date.toLocaleTimeString(
        [],
        {
            hour: "2-digit",
            minute: "2-digit"
        }
    );
}


function shortenId(value) {
    if (!value) {
        return "-";
    }

    if (value.length <= 12) {
        return value;
    }

    return `${value.substring(0, 8)}...`;
}


async function loadDashboard() {
    hideError();

    const hours = hoursSelect.value;

    try {
        const summary =
            await fetchJSON(
                `/dashboard/api/summary?hours=${hours}`
            );

        setText(
            "totalRequests",
            summary.total_requests
        );

        setText(
            "completed",
            summary.completed
        );

        setText(
            "inputBlocked",
            summary.input_blocked
        );

        setText(
            "outputBlocked",
            summary.output_blocked
        );

        setText(
            "llmErrors",
            summary.llm_errors
        );

        setText(
            "averageRisk",
            formatRisk(
                summary.average_overall_risk
            )
        );

        setText(
            "averageLatency",
            formatLatency(
                summary.average_latency_ms
            )
        );

        renderDistribution(
            "riskDistribution",
            summary.risk_level_distribution
        );

        renderDistribution(
            "inputDecisionDistribution",
            summary.input_decision_distribution
        );

        renderDistribution(
            "outputSafetyDistribution",
            summary.output_safety_distribution
        );

        renderDistribution(
            "languageDistribution",
            summary.language_distribution
        );

        renderTopCategories(
            "attackCategories",
            summary.top_attack_categories
        );

        renderTopCategories(
            "outputCategories",
            summary.top_output_safety_categories
        );


        const events =
            await fetchJSON(
                "/dashboard/api/events?limit=50"
            );

        renderEvents(events);


        const trends =
            await fetchJSON(
                `/dashboard/api/trends?hours=${hours}&bucket=hour`
            );

        renderTrends(trends);

    } catch (error) {
        console.error(error);

        showError(
            error.message ||
            "Unable to load dashboard data."
        );
    }
}


refreshButton.addEventListener(
    "click",
    loadDashboard
);


hoursSelect.addEventListener(
    "change",
    loadDashboard
);


loadDashboard();