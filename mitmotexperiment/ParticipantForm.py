from psychopy import core, gui
import re

class ParticipantForm:
    def __init__(self, win):
        self.win = win
        self.form_data = False

    def show_form(self):
        while True:
            # Create a GUI form
            form = gui.Dlg(title="Dane uczestnika")
            form.addText("Wprowadź swoje dane:")
            form.addField("Email:", "", required=True)  # Pre-fill email
            form.addField("Płeć:", choices=["F", "M"], required=True)  # Drop-down menu instead of text
            form.addField("Imię:", "", required=True)  # Pre-fill first name
            form.addField("Nazwisko:", "", required=True)  # Pre-fill last name

            # Show the form
            self.form_data = form.show()
            
            # Check if user pressed Cancel or closed the window
            if self.form_data is None:
                core.quit()

            # Assign form values (strip spaces)

            # Ensure all fields are filled
            if not all(list(self.form_data.values())):
                self.show_error("Wszystkie pola muszą być uzupełnione.")
                continue

            break  # Exit loop when everything is valid