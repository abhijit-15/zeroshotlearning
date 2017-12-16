$(document).ready(function() {
  console.log("ready!");

  $('#try-again').hide();

  // on form submission ...
  $('form').on('submit', function() {

    console.log("the form has beeen submitted");

    // grab values
    valueOne = $('input[name="url"]').val();
    console.log(valueOne);

    $.ajax({
      type: "POST",
      url: "/",
      data : { 'first': valueOne },
      success: function(results) {
        
          document.getElementById("results").innerHTML = 'results';
          console.log("Results are");
          console.log(results);

          } else {
          $('#results').html('Something went terribly wrong! Please try again.')
        }
      },
      error: function(error) {
        console.log(error)
      }
    });

  });

  $('#try-again').on('click', function(){
    $('input').val('').show();
    $('#try-again').hide();
    $('#results').html('');
  });

});