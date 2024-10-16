// Importing modules
import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
	function handleSubmit(e) {
		// Prevent the browser from reloading the page
		e.preventDefault();
		// Read the form data
		const form = e.target;
		const formData = new FormData(form);
		// You can pass formData as a fetch body directly:
		fetch('/optimize', { method: form.method, body: formData });
		// You can generate a URL out of it, as the browser does by default:
		console.log(new URLSearchParams(formData).toString());
		// You can work with it as a plain object.
		console.log([...formData.entries()]);
	  }

	const [data, setData] = useState([]);
	// Using useEffect for single rendering
	useEffect(() => {
		fetch('/uploads').then(res => res.json()).then(data => {
			console.log(data)
			setData(data);
		});
	  }, []);

	return (
		<div className="App">
			<div className="App-header">
				<h1 >WebBO: Multi-fidelity Bayesian Optimization</h1>
			</div>
			<div className="column">
				<h2>Submission</h2>
				<div className="File-upload">
					<p>I am hungry. Give me some files!</p>	
					<form method="POST" action="/upload" encType="multipart/form-data">
						<p><input type="file" name="file"></input></p>
						<p><input type="submit" value="Upload"></input></p>
					</form>
				</div>
				<div className="Form-submit">
					<form method="POST" onSubmit={handleSubmit}>
					<p>
						<label>
						Choose your Acquisition Function: 
						<select name="acquisition" defaultValue="mes">
						<option value="mes">MF-MES</option>
						<option value="tvr">MF-TVR</option>
						<option value="custom">MF-Custom</option>
						</select>
						</label>
					</p>
					<p>
						<label>
							Set a budget: 
						<input name="budget"></input>
						</label>
					</p>
					
					<p>
						<label>
						Choose the data to optimize: 
						{
						<select name="file">
							{ data.map(function(item) { return(<option value={item}>{item}</option>)}) }
						</select>		
						}
						</label>
					</p>
					<hr />
					<button type="submit">Optimize</button>
					</form>
				</div>
			</div>
			<div className="column">
				<h2>Results</h2>
			</div>
		</div>
	);
}

export default App;
