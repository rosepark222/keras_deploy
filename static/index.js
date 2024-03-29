(function()
{
	var canvas = document.querySelector( "#canvas" );
	var context = canvas.getContext( "2d" );
	canvas.width = 840;
	canvas.height = 280;

	console.log(`We are under ${this} context`);

	var Mouse = { x: 0, y: 0 };
	var lastMouse = { x: 0, y: 0 };
	var recordings = new Array(); //array of Recording objects

	context.fillStyle="white";
	context.fillRect(0,0,canvas.width,canvas.height);
	context.color = "black";
	context.lineWidth = 5;
    context.lineJoin = context.lineCap = 'round';
	
	debug();

	canvas.addEventListener( "mousemove", function( e )
	{
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

		Mouse.x = e.pageX - this.offsetLeft;
		Mouse.y = e.pageY - this.offsetTop;

	}, false );

	canvas.addEventListener( "mousedown", function( e )
	{
		canvas.addEventListener( "mousemove", onPaint, false );
 		recordings.push(-999);
	}, false );

	canvas.addEventListener( "mouseup", function()
	{
		canvas.removeEventListener( "mousemove", onPaint, false );

		/* canvas.addEventListener( "mouseup", mouseup_f, false ); */

	}, false );

	var mouseup_f = function()
	{
		recordings = new Array();
		console.log(`mouseup_f are under ${this} context`);

	}

	var onPaint = function()
	{	
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;
	
		context.beginPath();
		context.moveTo( lastMouse.x, lastMouse.y );
		context.lineTo( Mouse.x, Mouse.y );
		context.closePath();
		context.stroke();
		/* console.log(`onPaint are under ${this} context`); */


		var data = lastMouse.x + lastMouse.y;
		recordings.push(lastMouse.x);
		recordings.push(lastMouse.y);
		/* $('#result').text(' stroke '+ recordings); */


		let soMany = 10;
        console.log(`This is ${soMany} times easier!`);
/*		var canvasX = $(self.canvas).offset().left;
		var canvasY = $(self.canvas).offset().top;
		
		self.mouseDown = true;
		var x = Math.floor(event.pageX - canvasX);
		var y = Math.floor(event.pageY - canvasY);
		
		var	currAction = new Point(x,y,0);
		self.drawAction(currAction,true);
		if (self.currentRecording != null)
			self.currentRecording.addAction(currAction);
		event.preventDefault();
		return false;
*/
	};




	function debug()
	{
		/* Remove BUTTON */
		var RemoveButton = $( "#RemoveButton" );
		RemoveButton.on( "click", function()
		{
				   		//callme();

			   			//console.log(recordings);
			   			//console.log(recordings.length);
			   			recordings = new Array();
				context.clearRect( 0, 0, 280, 280 );
				context.fillStyle="white";
				context.fillRect(0,0,canvas.width,canvas.height);
			    $('#result').text('');
		});
		/*   BUTTON */
		var PredictButton = $( "#PredictButton1" );
		PredictButton.on( "click", function()
		{
				   		callme();

			   			console.log(recordings);
			   			console.log(recordings.length);

	   			var myJSON = JSON.stringify( recordings); 
	   			$.ajax({
	   				type: "POST",
	   				//url: $SCRIPT_ROOT + "/predict2/",
	   				url: "/predict1/",
	   				data: myJSON,
	   				success: function(data){
	   					$('#result_debug').text(data);
	   				}
	   			});
			
		});

		/*   BUTTON */
		var PredictButton15 = $( "#PredictButton2" );
		PredictButton15.on( "click", function()
		{
				   		callme();

			   			console.log(recordings);
			   			console.log(recordings.length);

	   			var myJSON = JSON.stringify( recordings); 
	   			$.ajax({
	   				type: "POST",
	   				//url: $SCRIPT_ROOT + "/predict2/",
	   				url: "/predict2/",
	   				data: myJSON,
	   				success: function(data){
	   					$('#result_debug').text(data);
	   					MathJax.Hub.Queue(["Typeset",MathJax.Hub,"result_debug"]);
	   					
	   				}
	   			});
			
		});	

		/*   BUTTON */
		var PredictButton53 = $( "#PredictButton3" );
		PredictButton53.on( "click", function()
		{
				   		callme();

			   			console.log(recordings);
			   			console.log(recordings.length);

	   			var myJSON = JSON.stringify( recordings); 
	   			$.ajax({
	   				type: "POST",
	   				url: "/predict3/",
	   				data: myJSON,
	   				success: function(data){
	   					// $('#result_math').text(" $$x = { -b + \\sqrt{b^2-4ac} \\over 2a}.$$ ");
	   					$('#result_math').text(data);
	   					MathJax.Hub.Queue(["Typeset",MathJax.Hub,"result_math"]);
	   					/* https://stackoverflow.com/questions/8151318/mathjax-not-working-in-ajax-based-web-page
	   					http://docs.mathjax.org/en/latest/advanced/typeset.html
	   					*/
	   				}
	   			});
			
		});					
		/* COLOR SELECTOR */

		$( "#colors" ).change(function()
		{
			var color = $( "#colors" ).val();
			context.color = color;
		});
		
		/* LINE WIDTH */
		
		$( "#lineWidth" ).change(function()
		{
			context.lineWidth = $( this ).val();
		});



		$(".myButton2").click(function(){
	   			//var $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};
	   			//var canvasObj = document.getElementById("canvas");
	   			//var img = '3'; 
	   			//var buffer = new Array();
	   			//buffer.push('4')
	   			//buffer.push('5')
	   			var myJSON = JSON.stringify( recordings); 
	   			
	   			$.ajax({
	   				type: "POST",
	   				//url: $SCRIPT_ROOT + "/predict2/",
	   				url: "/predict2/",
	   				data: myJSON,
	   				success: function(data){
	   					$('#result').text(' Predicted Output: '+data);
	   				}
	   			});
	   		});


	}
}());


function callme() {
	console.log(`callme was called`);
}
