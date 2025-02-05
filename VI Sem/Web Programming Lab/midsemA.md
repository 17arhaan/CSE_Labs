
---

## **🟢 Lab 1: HTML5**
### **1. What is the purpose of `<!DOCTYPE html>` in HTML5?**
- The `<!DOCTYPE html>` declaration tells the browser that the document is written in **HTML5**.  
- It helps browsers render the page correctly.

### **2. Explain the difference between semantic and non-semantic elements in HTML5. Provide examples.**
- **Semantic elements**: Clearly define their meaning in a human and machine-readable way.
  - Examples: `<header>`, `<article>`, `<section>`, `<footer>`, `<nav>`
- **Non-semantic elements**: Do not have a meaningful name.
  - Examples: `<div>`, `<span>`

### **3. What are the new input types introduced in HTML5? Provide at least three examples.**
- `type="email"` → Validates email addresses.
- `type="date"` → Allows users to select a date.
- `type="number"` → Accepts only numerical input.

### **4. Write an HTML5 form with Name, Email, Password, and Date of Birth fields.**
```html
<form>
    <label>Name:</label>
    <input type="text" name="name"><br>

    <label>Email:</label>
    <input type="email" name="email"><br>

    <label>Password:</label>
    <input type="password" name="password"><br>

    <label>Date of Birth:</label>
    <input type="date" name="dob"><br>

    <button type="submit">Submit</button>
</form>
```

### **5. What is the `<canvas>` element used for?**
- It is used for **drawing graphics** via JavaScript.

**Example to draw a rectangle:**
```html
<canvas id="myCanvas" width="200" height="100" style="border:1px solid black;"></canvas>
<script>
    var canvas = document.getElementById("myCanvas");
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "blue";
    ctx.fillRect(20, 20, 150, 80);
</script>
```

### **6. Explain the difference between `<video>` and `<audio>` elements.**
- `<video>` is used to embed videos.
- `<audio>` is used to embed sound files.

**Example:**
```html
<video width="320" height="240" controls>
    <source src="video.mp4" type="video/mp4">
</video>

<audio controls>
    <source src="audio.mp3" type="audio/mpeg">
</audio>
```

### **7. What is `localStorage` and `sessionStorage`?**
| Feature | `localStorage` | `sessionStorage` |
|---------|---------------|------------------|
| Data Lifetime | Permanent | Until the session ends |
| Storage Limit | 5MB | 5MB |
| Data Shared Between Tabs? | Yes | No |

### **8. What is the difference between `<iframe>` and `<embed>`?**
- **`<iframe>`** → Used to embed an external webpage.
- **`<embed>`** → Used for embedding multimedia.

### **9. What are HTML5 APIs? Mention two with examples.**
- **Geolocation API** → Gets user’s location.
```js
navigator.geolocation.getCurrentPosition(function(position) {
    alert(position.coords.latitude + ", " + position.coords.longitude);
});
```
- **Drag and Drop API** → Enables dragging elements.

### **10. How do you create a progress bar in HTML5?**
```html
<progress value="50" max="100"></progress>
```

---

## **🟢 Lab 2: CSS & Bootstrap**
### **CSS Questions**
### **1. Explain the three ways to apply CSS to an HTML document.**
- **Inline CSS** → Inside an HTML tag.  
  ```html
  <p style="color:red;">Hello</p>
  ```
- **Internal CSS** → Inside `<style>` in the `<head>`.  
  ```html
  <style> p { color: blue; } </style>
  ```
- **External CSS** → In a separate `.css` file.  
  ```css
  p { color: green; }
  ```

### **2. Difference between `id` and `class` selectors in CSS**
- **`id`**: Unique (used for one element).
- **`class`**: Reusable (can be used for multiple elements).

**Example:**
```css
#uniqueId { color: red; }
.className { color: blue; }
```

### **3. Explain the CSS Box Model.**
- **Components:** `margin`, `border`, `padding`, `content`.

### **4. Write a CSS rule to style a paragraph.**
```css
p {
    font-size: 18px;
    color: blue;
    background-color: lightgray;
    text-align: center;
}
```

### **5. Difference between `relative`, `absolute`, `fixed`, and `sticky` positioning.**
| Type | Description |
|------|-------------|
| Relative | Moves relative to its normal position. |
| Absolute | Positioned relative to the nearest positioned ancestor. |
| Fixed | Stays in place even when scrolling. |
| Sticky | Moves between `relative` and `fixed` based on scroll. |

---

## **Bootstrap Questions**
### **1. What is Bootstrap?**
- Bootstrap is a **CSS framework** for responsive web design.

### **2. How do you include Bootstrap in an HTML file?**
```html
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
```

### **3. Difference between `.container` and `.container-fluid`?**
- `.container`: Fixed width.
- `.container-fluid`: Full width.

### **4. Example of Bootstrap Grid System**
```html
<div class="row">
    <div class="col-md-6">Column 1</div>
    <div class="col-md-6">Column 2</div>
</div>
```

### **5. Example of a Bootstrap Navigation Bar**
```html
<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <a class="navbar-brand" href="#">My Site</a>
</nav>
```

---

## **🟢 Lab 3: JavaScript**
### **1. Difference between `var`, `let`, and `const`**
| Type | Scope | Reassignable? |
|------|--------|--------------|
| var | Function scope | Yes |
| let | Block scope | Yes |
| const | Block scope | No |

### **2. JavaScript function to sum an array**
```js
function sumArray(arr) {
    return arr.reduce((a, b) => a + b, 0);
}
```

### **3. Example of `onclick` event**
```html
<button onclick="alert('Hello!')">Click Me</button>
```

---

## **🟢 Lab 4: jQuery**
### **1. What is jQuery?**
- jQuery is a **JavaScript library** that simplifies DOM manipulation.

### **2. Selecting elements using class and ID in jQuery**
```js
$(".className").hide(); // Hide all elements with class
$("#idName").show(); // Show an element with an ID
```

### **3. jQuery to hide an element**
```js
$("#btn").click(function() {
    $("#element").hide();
});
```

### **4. Animate a div moving right**
```js
$("#box").animate({left: '+=200px'}, 500);
```

---
