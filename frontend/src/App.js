// Importing modules
import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
	// usestate for setting a javascript
	// object for storing and using data
	const [data, setdata] = useState({
		name: "",
		age: 0,
		date: "",
		programming: "",
	});

	// Using useEffect for single rendering
	useEffect(() => {
		// Using fetch to fetch the api from 
		// flask server it will be redirected to proxy
		fetch("/data").then((res) =>
			res.json().then((data) => {
				// Setting a data from api
				setdata({
					name: data.Name,
					age: data.Age,
					date: data.Date,
					programming: data.programming,
				});
			})
		);
	}, []);

	return (
		<div className="App">
			<header className="App-header">
				<h3>Multi-fidelity Bayesian Optimization</h3>
			</header>
			{/* <p>{data.name}</p>
				<p>{data.age}</p>
				<p>{data.date}</p>
				<p>{data.programming}</p> */}
			<form method="POST" action="/upload" encType="multipart/form-data">
      			<p><input type="file" name="file"></input></p>
      			<p><input type="submit" value="Submit"></input></p>
    		</form>
		</div>
	);
}

export default App;
