document.addEventListener("DOMContentLoaded", function() {
    // Set up today's date for highlighting
    const today = new Date();
    let currentMonth = today.getMonth();
    let currentYear = today.getFullYear();

    const monthNames = [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ];

    function renderCalendar(month, year) {
        const calendarBody = document.getElementById("calendar-body");
        calendarBody.innerHTML = "";

        // Set header month and year
        document.getElementById("monthYear").innerText = monthNames[month] + " " + year;

        // Get first day (0 = Sunday, 1 = Monday, etc.)
        const firstDay = new Date(year, month, 1).getDay();
        // Get number of days in month
        const daysInMonth = new Date(year, month + 1, 0).getDate();

        let date = 1;
        // Create 6 rows (weeks) for the calendar
        for (let i = 0; i < 6; i++) {
            const row = document.createElement("tr");

            // Create 7 columns for each day of the week
            for (let j = 0; j < 7; j++) {
                const cell = document.createElement("td");
                if (i === 0 && j < firstDay) {
                    // Empty cell before the first day of the month
                    cell.innerText = "";
                } else if (date > daysInMonth) {
                    // Empty cells after the last date
                    cell.innerText = "";
                } else {
                    cell.innerText = date;
                    // Highlight today's date if it matches
                    if (year === today.getFullYear() && month === today.getMonth() && date === today.getDate()) {
                        cell.classList.add("today");
                    }
                    date++;
                }
                row.appendChild(cell);
            }
            calendarBody.appendChild(row);
        }
    }

    // Button event listeners for navigation
    document.getElementById("prevBtn").addEventListener("click", function() {
        currentMonth--;
        if (currentMonth < 0) {
            currentMonth = 11;
            currentYear--;
        }
        renderCalendar(currentMonth, currentYear);
    });

    document.getElementById("nextBtn").addEventListener("click", function() {
        currentMonth++;
        if (currentMonth > 11) {
            currentMonth = 0;
            currentYear++;
        }
        renderCalendar(currentMonth, currentYear);
    });

    // Render the initial calendar view
    renderCalendar(currentMonth, currentYear);
});
